#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef uint64_t u64;

typedef struct
{
    int n;
    size_t S;    
    size_t rows; 
    u64 *X;
    u64 *Z;
    unsigned char *sign;
    unsigned char *flag; 
    int *idx;
} Tableau;

static u64 rng_state = 88172645463325252ULL;

static u64 next_random(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void die(const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

static inline u64 *XR(const Tableau *T, size_t i) { return T->X + i * T->S; }
static inline u64 *ZR(const Tableau *T, size_t i) { return T->Z + i * T->S; }

static void *alloc_aligned(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0)
        die("Memory allocation failed.");
    return p;
}

static Tableau *create_tableau(int n)
{
    Tableau *T = malloc(sizeof(Tableau));
    if (!T)
        die("Memory allocation failed.");

    T->n = n;
    T->S = (((size_t)n + 63) / 64 + 7) / 8 * 8;
    T->rows = 2 * (size_t)n;

    size_t bytes = T->rows * T->S * sizeof(u64);
    T->X = alloc_aligned(bytes);
    T->Z = alloc_aligned(bytes);
    T->sign = malloc(T->rows);
    T->flag = malloc(T->rows);
    T->idx = malloc((size_t)n * sizeof(int));
    if (!T->sign || !T->flag || !T->idx)
        die("Memory allocation failed.");
    return T;
}

static void initialize_tableau(Tableau *T)
{
    int n = T->n;
    long rows = (long)T->rows;

    #pragma omp parallel for schedule(static) if(n >= 256)
    for (long i = 0; i < rows; i++)
    {
        memset(XR(T, i), 0, T->S * sizeof(u64));
        memset(ZR(T, i), 0, T->S * sizeof(u64));
        T->sign[i] = 0;
    }

    for (int i = 0; i < n; i++)
    {
        XR(T, i)[i >> 6] |= 1ULL << (i & 63);
        ZR(T, n + i)[i >> 6] |= 1ULL << (i & 63);
    }
}

static void free_tableau(Tableau *T)
{
    if (!T)
        return;
    free(T->X);
    free(T->Z);
    free(T->sign);
    free(T->flag);
    free(T->idx);
    free(T);
}



static inline unsigned getb(const u64 *r, int q)
{
    return (unsigned)((r[q >> 6] >> (q & 63)) & 1ULL);
}

static inline void flipb(u64 *r, int q, unsigned v)
{
    r[q >> 6] ^= (u64)v << (q & 63);
}

static inline unsigned row_H(u64 *x, u64 *z, unsigned s, int q)
{
    unsigned xa = getb(x, q), za = getb(z, q);
    s ^= xa & za;
    unsigned d = xa ^ za;
    flipb(x, q, d);
    flipb(z, q, d);
    return s;
}

static inline unsigned row_P(u64 *x, u64 *z, unsigned s, int q)
{
    unsigned xa = getb(x, q), za = getb(z, q);
    s ^= xa & za;
    flipb(z, q, xa);
    return s;
}

static inline unsigned row_CX(u64 *x, u64 *z, unsigned s, int c, int t)
{
    unsigned xc = getb(x, c), zc = getb(z, c);
    unsigned xt = getb(x, t), zt = getb(z, t);
    s ^= xc & zt & (xt ^ zc ^ 1u);
    flipb(z, c, zt);
    flipb(x, t, xc);
    return s;
}

static void apply_layer(Tableau *T, const int *perm)
{
    int n = T->n;
    long rows = (long)T->rows;

    #pragma omp parallel for schedule(static) if(n >= 64)
    for (long i = 0; i < rows; i++)
    {
        u64 *x = XR(T, i);
        u64 *z = ZR(T, i);
        unsigned s = T->sign[i];

        for (int k = 0; k + 1 < n; k += 2)
        {
            int a = perm[k], b = perm[k + 1];
            s = row_H(x, z, s, a);
            s = row_P(x, z, s, b);
            s = row_CX(x, z, s, a, b);
        }
        T->sign[i] = (unsigned char)s;
    }
}

enum { G_H = 0, G_P = 1, G_CX = 2 };

static void apply_gate(Tableau *T, int type, int a, int b)
{
    long rows = (long)T->rows;

    #pragma omp parallel for schedule(static) if(T->n >= 256)
    for (long i = 0; i < rows; i++)
    {
        u64 *x = XR(T, i), *z = ZR(T, i);
        unsigned s = T->sign[i];
        if (type == G_H)
            s = row_H(x, z, s, a);
        else if (type == G_P)
            s = row_P(x, z, s, a);
        else
            s = row_CX(x, z, s, a, b);
        T->sign[i] = (unsigned char)s;
    }
}

static inline int word_phase(u64 x1, u64 z1, u64 x2, u64 z2)
{
    u64 plus  = (x1 & z1 & ~x2 & z2) | (x1 & ~z1 & x2 & z2) | (~x1 & z1 & x2 & ~z2);
    u64 minus = (x1 & z1 & x2 & ~z2) | (x1 & ~z1 & ~x2 & z2) | (~x1 & z1 & x2 & z2);
    return __builtin_popcountll(plus) - __builtin_popcountll(minus);
}

static inline void rowmult(Tableau *T, size_t h, size_t s)
{
    long long ph = 2LL * (T->sign[h] + T->sign[s]);
    u64 *xh = XR(T, h), *zh = ZR(T, h);
    const u64 *xs = XR(T, s), *zs = ZR(T, s);

    for (size_t w = 0; w < T->S; w++)
    {
        u64 a = xs[w], b = zs[w], c = xh[w], d = zh[w];
        ph += word_phase(a, b, c, d);
        xh[w] = a ^ c;
        zh[w] = b ^ d;
    }
    T->sign[h] = ((ph & 3) == 2);
}



static int measure_qubit(Tableau *T, int q)
{
    int n = T->n;
    long rows = (long)T->rows;
    size_t S = T->S;
    size_t w = (size_t)q >> 6;
    int b = q & 63;
    unsigned char *flag = T->flag;

    #pragma omp parallel for schedule(static) if(n >= 256)
    for (long i = 0; i < rows; i++)
        flag[i] = (unsigned char)((XR(T, i)[w] >> b) & 1ULL);

    const unsigned char *hit = memchr(flag + n, 1, (size_t)n);

    if (hit)
    {
      
        long p = hit - flag;

        #pragma omp parallel for schedule(static) if(n >= 256)
        for (long i = 0; i < rows; i++)
            if (flag[i] && i != p)
                rowmult(T, (size_t)i, (size_t)p);

        long d = p - n;
        memcpy(XR(T, d), XR(T, p), S * sizeof(u64));
        memcpy(ZR(T, d), ZR(T, p), S * sizeof(u64));
        T->sign[d] = T->sign[p];

        memset(XR(T, p), 0, S * sizeof(u64));
        memset(ZR(T, p), 0, S * sizeof(u64));
        ZR(T, p)[w] |= 1ULL << b;

        int result = (int)((next_random() >> 33) & 1ULL);
        T->sign[p] = (unsigned char)result;
        return result;
    }

    int k = 0;
    for (int i = 0; i < n; i++)
        if (flag[i])
            T->idx[k++] = n + i;

    long long ph = 0;
    for (int j = 0; j < k; j++)
        ph += 2LL * T->sign[T->idx[j]];

    long nblk = (long)(S / 8);
    const int *idx = T->idx;

    #pragma omp parallel for reduction(+:ph) schedule(static) if((size_t)k * S >= 16384)
    for (long blk = 0; blk < nblk; blk++)
    {
        u64 ax[8] = {0}, az[8] = {0};
        long long loc = 0;
        size_t off = (size_t)blk * 8;

        for (int j = 0; j < k; j++)
        {
            const u64 *xs = XR(T, (size_t)idx[j]) + off;
            const u64 *zs = ZR(T, (size_t)idx[j]) + off;
            for (int t = 0; t < 8; t++)
            {
                loc += word_phase(xs[t], zs[t], ax[t], az[t]);
                ax[t] ^= xs[t];
                az[t] ^= zs[t];
            }
        }
        ph += loc;
    }

    return ((ph & 3) == 2);
}



static int test_bell(void)
{
    for (int trial = 0; trial < 200; trial++)
    {
        Tableau *T = create_tableau(2);
        initialize_tableau(T);
        apply_gate(T, G_H, 0, 0);
        apply_gate(T, G_CX, 0, 1);
        int a = measure_qubit(T, 0);
        int b = measure_qubit(T, 1);
        free_tableau(T);
        if (a != b)
            return 0;
    }
    return 1;
}

static int test_ghz(int n)
{
    for (int trial = 0; trial < 50; trial++)
    {
        Tableau *T = create_tableau(n);
        initialize_tableau(T);
        apply_gate(T, G_H, 0, 0);
        for (int q = 1; q < n; q++)
            apply_gate(T, G_CX, 0, q);

        int first = measure_qubit(T, 0), ok = 1;
        for (int q = 1; q < n; q++)
            if (measure_qubit(T, q) != first)
                ok = 0;
        free_tableau(T);
        if (!ok)
            return 0;
    }
    return 1;
}

static int test_repeatability(int n)
{
    Tableau *T = create_tableau(n);
    initialize_tableau(T);
    for (int q = 0; q < n; q++)
        apply_gate(T, G_H, q, 0);

    int ok = 1;
    for (int q = 0; q < n && ok; q++)
    {
        int r1 = measure_qubit(T, q);
        int r2 = measure_qubit(T, q);
        if (r1 != r2)
            ok = 0;
    }
    free_tableau(T);
    return ok;
}


static int test_inverse(int n, int ngates)
{
    Tableau *T = create_tableau(n);
    initialize_tableau(T);

    int *ty = malloc(ngates * sizeof(int));
    int *qa = malloc(ngates * sizeof(int));
    int *qb = malloc(ngates * sizeof(int));

    for (int g = 0; g < ngates; g++)
    {
        ty[g] = (int)(next_random() % 3);
        qa[g] = (int)(next_random() % (u64)n);
        qb[g] = 0;
        if (ty[g] == G_CX)
        {
            do
                qb[g] = (int)(next_random() % (u64)n);
            while (qb[g] == qa[g]);
        }
        apply_gate(T, ty[g], qa[g], qb[g]);
    }

    for (int g = ngates - 1; g >= 0; g--)
    {
        int reps = (ty[g] == G_P) ? 3 : 1;   // P^-1 = P^3
        for (int r = 0; r < reps; r++)
            apply_gate(T, ty[g], qa[g], qb[g]);
    }

    int ok = 1;
    for (int q = 0; q < n; q++)
        if (measure_qubit(T, q) != 0)
            ok = 0;

    free(ty);
    free(qa);
    free(qb);
    free_tableau(T);
    return ok;
}

static int sympl(const Tableau *T, size_t i, size_t j)
{
    int par = 0;
    for (size_t w = 0; w < T->S; w++)
        par ^= __builtin_popcountll((XR(T, i)[w] & ZR(T, j)[w]) ^
                                    (ZR(T, i)[w] & XR(T, j)[w])) & 1;
    return par;
}

static int test_invariants(int n, int ngates)
{
    Tableau *T = create_tableau(n);
    initialize_tableau(T);

    for (int g = 0; g < ngates; g++)
    {
        int t = (int)(next_random() % 3);
        int a = (int)(next_random() % (u64)n);
        int b = 0;
        if (t == G_CX)
        {
            do
                b = (int)(next_random() % (u64)n);
            while (b == a);
        }
        apply_gate(T, t, a, b);
        if (g % 25 == 0)
            measure_qubit(T, (int)(next_random() % (u64)n));
    }

    int ok = 1;
    for (int i = 0; i < n && ok; i++)
        for (int j = 0; j < n && ok; j++)
        {
            if (sympl(T, n + i, n + j) != 0) ok = 0;
            if (sympl(T, i, j) != 0) ok = 0;
            if (sympl(T, i, n + j) != (i == j)) ok = 0;
        }

    free_tableau(T);
    return ok;
}

static void run_tests(void)
{
    struct { const char *name; int pass; } r[] = {
        {"bell",                      test_bell()},
        {"ghz (8)",                   test_ghz(8)},
        {"ghz (200, multi-word)",     test_ghz(200)},
        {"repeatability (130)",       test_repeatability(130)},
        {"circuit + inverse (70)",    test_inverse(70, 2000)},
        {"circuit + inverse (300)",   test_inverse(300, 3000)},
        {"invariants (130)",          test_invariants(130, 1500)},
        {"invariants (300)",          test_invariants(300, 1500)},
    };

    int all = 1;
    for (size_t i = 0; i < sizeof r / sizeof r[0]; i++)
    {
        printf("%-28s %s\n", r[i].name, r[i].pass ? "ok" : "FAILED");
        all &= r[i].pass;
    }
    printf(all ? "All correctness tests passed.\n" : "Correctness test failed.\n");
}


static void run_benchmark(int n, int depth, int measure_every)
{
    Tableau *T = create_tableau(n);
    initialize_tableau(T);

    int *perm = malloc((size_t)n * sizeof(int));
    if (!perm)
        die("Memory allocation failed.");
    for (int i = 0; i < n; i++)
        perm[i] = i;

    double gate_time = 0.0, meas_time = 0.0;
    long long gate_count = 0, meas_count = 0;

    for (int d = 0; d < depth; d++)
    {
        for (int i = n - 1; i > 0; i--)
        {
            int j = (int)(next_random() % (u64)(i + 1));
            int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        }

        double t0 = now_seconds();
        apply_layer(T, perm);
        gate_time += now_seconds() - t0;
        gate_count += 3LL * (n / 2);

        if ((d + 1) % measure_every == 0)
        {
            double t1 = now_seconds();
            for (int q = 0; q < n; q += 7)
            {
                measure_qubit(T, q);
                meas_count++;
            }
            meas_time += now_seconds() - t1;
        }
    }

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    printf("Threads             : %d\n", threads);
    printf("Qubits              : %d\n", n);
    printf("Depth               : %d\n", depth);
    printf("Gate operations     : %lld\n", gate_count);
    printf("Measurements        : %lld\n\n", meas_count);
    printf("Gate time           : %.6f s\n", gate_time);
    printf("Measurement time    : %.6f s (pivot search included)\n", meas_time);
    printf("Total measured time : %.6f s\n", gate_time + meas_time);

    free(perm);
    free_tableau(T);
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--test") == 0)
    {
        run_tests();
        return EXIT_SUCCESS;
    }

    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <qubits> <depth> <measure_every>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    int depth = atoi(argv[2]);
    int measure_every = atoi(argv[3]);

    if (n <= 0 || depth <= 0 || measure_every <= 0)
    {
        fprintf(stderr, "Invalid arguments.\n");
        return EXIT_FAILURE;
    }

    run_benchmark(n, depth, measure_every);
    return EXIT_SUCCESS;
}