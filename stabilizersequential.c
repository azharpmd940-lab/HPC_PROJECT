#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct
{
    int n;
    unsigned char **x;
    unsigned char **z;
    unsigned char *sign;
} Tableau;

static unsigned long long rng_state =
    88172645463325252ULL;

static unsigned long long next_random(void)
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

    return (double)ts.tv_sec +
           (double)ts.tv_nsec * 1e-9;
}

static Tableau *create_tableau(int n)
{
    Tableau *T = malloc(sizeof(Tableau));

    if (T == NULL)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    T->n = n;

    int rows = 2 * n + 1;

    T->x = malloc(rows * sizeof(unsigned char *));
    T->z = malloc(rows * sizeof(unsigned char *));
    T->sign = malloc(rows * sizeof(unsigned char));

    if (T->x == NULL ||
        T->z == NULL ||
        T->sign == NULL)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++)
    {
        T->x[i] = calloc(n, sizeof(unsigned char));
        T->z[i] = calloc(n, sizeof(unsigned char));

        if (T->x[i] == NULL ||
            T->z[i] == NULL)
        {
            fprintf(stderr, "Memory allocation failed.\n");
            exit(EXIT_FAILURE);
        }

        T->sign[i] = 0;
    }

    return T;
}

static void initialize_tableau(Tableau *T)
{
    int n = T->n;

    for (int i = 0; i < 2 * n + 1; i++)
    {
        memset(T->x[i], 0, n);
        memset(T->z[i], 0, n);
        T->sign[i] = 0;
    }

    for (int i = 0; i < n; i++)
    {
        T->x[i][i] = 1;
        T->z[n + i][i] = 1;
    }
}

static void free_tableau(Tableau *T)
{
    if (T == NULL)
        return;

    int rows = 2 * T->n + 1;

    for (int i = 0; i < rows; i++)
    {
        free(T->x[i]);
        free(T->z[i]);
    }

    free(T->x);
    free(T->z);
    free(T->sign);
    free(T);
}

static void apply_H(Tableau *T, int q)
{
    int rows = 2 * T->n;

    for (int i = 0; i < rows; i++)
    {
        if (T->x[i][q] && T->z[i][q])
            T->sign[i] ^= 1;

        unsigned char temp = T->x[i][q];

        T->x[i][q] = T->z[i][q];
        T->z[i][q] = temp;
    }
}

static void apply_P(Tableau *T, int q)
{
    int rows = 2 * T->n;

    for (int i = 0; i < rows; i++)
    {
        if (T->x[i][q] && T->z[i][q])
            T->sign[i] ^= 1;

        T->z[i][q] ^= T->x[i][q];
    }
}

static void apply_CX(
    Tableau *T,
    int control,
    int target)
{
    int rows = 2 * T->n;

    for (int i = 0; i < rows; i++)
    {
        unsigned char xc = T->x[i][control];
        unsigned char zc = T->z[i][control];

        unsigned char xt = T->x[i][target];
        unsigned char zt = T->z[i][target];

        if (xc && zt &&
            (xt ^ zc ^ 1))
        {
            T->sign[i] ^= 1;
        }

        T->z[i][control] ^= zt;
        T->x[i][target] ^= xc;
    }
}

static int phase_exponent(
    unsigned char x1,
    unsigned char z1,
    unsigned char x2,
    unsigned char z2)
{
    if (x1 == 0 && z1 == 0)
        return 0;

    if (x1 == 1 && z1 == 1)
        return (int)z2 - (int)x2;

    if (x1 == 1)
        return (int)z2 * (2 * (int)x2 - 1);

    return (int)x2 * (1 - 2 * (int)z2);
}

static void multiply_rows(
    Tableau *T,
    int destination,
    int source)
{
    int n = T->n;
    int phase = 0;

    if (T->sign[destination])
        phase += 2;

    if (T->sign[source])
        phase += 2;

    for (int q = 0; q < n; q++)
    {
        phase += phase_exponent(
            T->x[destination][q],
            T->z[destination][q],
            T->x[source][q],
            T->z[source][q]);
    }

    phase = ((phase % 4) + 4) % 4;

    T->sign[destination] =
        (phase == 2);

    for (int q = 0; q < n; q++)
    {
        T->x[destination][q] ^=
            T->x[source][q];

        T->z[destination][q] ^=
            T->z[source][q];
    }
}

static int find_pivot(
    Tableau *T,
    int q)
{
    int n = T->n;

    for (int i = n; i < 2 * n; i++)
    {
        if (T->x[i][q])
            return i;
    }

    return -1;
}

static int measure_qubit(
    Tableau *T,
    int q,
    int pivot)
{
    int n = T->n;

    if (pivot == -1)
    {
        int scratch = 2 * n;

        memset(T->x[scratch], 0, n);
        memset(T->z[scratch], 0, n);

        T->sign[scratch] = 0;

        for (int i = 0; i < n; i++)
        {
            if (T->x[i][q])
            {
                multiply_rows(
                    T,
                    scratch,
                    n + i);
            }
        }

        return T->sign[scratch];
    }

    for (int i = 0; i < 2 * n; i++)
    {
        if (i != pivot &&
            T->x[i][q])
        {
            multiply_rows(
                T,
                i,
                pivot);
        }
    }

    int destabilizer = pivot - n;

    memcpy(
        T->x[destabilizer],
        T->x[pivot],
        n * sizeof(unsigned char));

    memcpy(
        T->z[destabilizer],
        T->z[pivot],
        n * sizeof(unsigned char));

    T->sign[destabilizer] =
        T->sign[pivot];

    memset(T->x[pivot], 0, n);
    memset(T->z[pivot], 0, n);

    T->z[pivot][q] = 1;

    int result =
        (int)(next_random() & 1ULL);

    T->sign[pivot] = result;

    return result;
}

static int test_bell(void)
{
    Tableau *T = create_tableau(2);

    initialize_tableau(T);

    apply_H(T, 0);
    apply_CX(T, 0, 1);

    int pivot0 = find_pivot(T, 0);
    int result0 = measure_qubit(T, 0, pivot0);

    int pivot1 = find_pivot(T, 1);
    int result1 = measure_qubit(T, 1, pivot1);

    free_tableau(T);

    return result0 == result1;
}

static int test_ghz(void)
{
    int n = 8;

    Tableau *T = create_tableau(n);

    initialize_tableau(T);

    apply_H(T, 0);

    for (int q = 1; q < n; q++)
        apply_CX(T, 0, q);

    int first = -1;
    int correct = 1;

    for (int q = 0; q < n; q++)
    {
        int pivot = find_pivot(T, q);
        int result = measure_qubit(T, q, pivot);

        if (q == 0)
            first = result;
        else if (result != first)
            correct = 0;
    }

    free_tableau(T);

    return correct;
}

static int test_repeatability(void)
{
    int n = 32;

    Tableau *T = create_tableau(n);

    initialize_tableau(T);

    int correct = 1;

    for (int q = 0; q < n; q++)
    {
        int pivot = find_pivot(T, q);
        int result = measure_qubit(T, q, pivot);

        if (result != 0)
        {
            correct = 0;
            break;
        }
    }

    free_tableau(T);

    return correct;
}

static void run_tests(void)
{
    int bell = test_bell();
    int ghz = test_ghz();
    int repeat = test_repeatability();

    if (bell && ghz && repeat)
        printf("All correctness tests passed.\n");
    else
        printf("Correctness test failed.\n");
}

static void run_benchmark(
    int n,
    int depth,
    int measure_every)
{
    Tableau *T = create_tableau(n);

    initialize_tableau(T);

    int *perm = malloc(n * sizeof(int));

    if (perm == NULL)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        free_tableau(T);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++)
        perm[i] = i;

    double gate_time = 0.0;
    double measurement_time = 0.0;
    double pivot_time = 0.0;

    long long gate_count = 0;
    long long measurement_count = 0;

    for (int d = 0; d < depth; d++)
    {
        for (int i = n - 1; i > 0; i--)
        {
            int j =
                (int)(next_random() %
                (unsigned long long)(i + 1));

            int temp = perm[i];
            perm[i] = perm[j];
            perm[j] = temp;
        }

        double start_gate = now_seconds();

        for (int i = 0; i + 1 < n; i += 2)
        {
            int q1 = perm[i];
            int q2 = perm[i + 1];

            apply_H(T, q1);
            apply_P(T, q2);
            apply_CX(T, q1, q2);

            gate_count += 3;
        }

        gate_time +=
            now_seconds() - start_gate;

        if ((d + 1) % measure_every == 0)
        {
            for (int q = 0; q < n; q += 7)
            {
                double start_pivot =
                    now_seconds();

                int pivot =
                    find_pivot(T, q);

                pivot_time +=
                    now_seconds() - start_pivot;

                double start_measure =
                    now_seconds();

                measure_qubit(
                    T,
                    q,
                    pivot);

                measurement_time +=
                    now_seconds() - start_measure;

                measurement_count++;
            }
        }
    }

    double total_time =
        gate_time +
        measurement_time +
        pivot_time;

    printf("Qubits              : %d\n", n);
    printf("Depth               : %d\n", depth);
    printf("Gate operations     : %lld\n", gate_count);
    printf("Measurements        : %lld\n", measurement_count);

    printf("\n");

    printf("Gate time           : %.6f s\n",
           gate_time);

    printf("Measurement time    : %.6f s\n",
           measurement_time);

    printf("Pivot search time   : %.6f s\n",
           pivot_time);

    printf("Total measured time : %.6f s\n",
           total_time);

    free(perm);
    free_tableau(T);
}

int main(int argc, char **argv)
{
    if (argc == 2 &&
        strcmp(argv[1], "--test") == 0)
    {
        run_tests();
        return EXIT_SUCCESS;
    }

    if (argc != 4)
    {
        fprintf(
            stderr,
            "Usage: %s <qubits> <depth> <measure_every>\n",
            argv[0]);

        return EXIT_FAILURE;
    }

    int n = atoi(argv[1]);
    int depth = atoi(argv[2]);
    int measure_every = atoi(argv[3]);

    if (n <= 0 ||
        depth <= 0 ||
        measure_every <= 0)
    {
        fprintf(stderr, "Invalid arguments.\n");
        return EXIT_FAILURE;
    }

    run_benchmark(
        n,
        depth,
        measure_every);

    return EXIT_SUCCESS;
}