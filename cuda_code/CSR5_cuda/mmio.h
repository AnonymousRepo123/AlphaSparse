/* 
*   Matrix Market I/O library for ANSI C
*
*   See http://math.nist.gov/MatrixMarket for details.
*
*
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#ifndef MM_IO_H
#define MM_IO_H

#include <sys/time.h>

#define MM_MAX_LINE_LENGTH 1025
#define MatrixMarketBanner "%%MatrixMarket"
#define MM_MAX_TOKEN_LENGTH 64

typedef char MM_typecode[4];

char *mm_typecode_to_str(MM_typecode matcode);

int mm_read_banner(FILE *f, MM_typecode *matcode);
int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz);
int mm_read_mtx_array_size(FILE *f, int *M, int *N);

int mm_write_banner(FILE *f, MM_typecode matcode);
int mm_write_mtx_crd_size(FILE *f, int M, int N, int nz);
int mm_write_mtx_array_size(FILE *f, int M, int N);


/********************* MM_typecode query fucntions ***************************/

#define mm_is_matrix(typecode)	((typecode)[0]=='M')

#define mm_is_sparse(typecode)	((typecode)[1]=='C')
#define mm_is_coordinate(typecode)((typecode)[1]=='C')
#define mm_is_dense(typecode)	((typecode)[1]=='A')
#define mm_is_array(typecode)	((typecode)[1]=='A')

#define mm_is_complex(typecode)	((typecode)[2]=='C')
#define mm_is_real(typecode)		((typecode)[2]=='R')
#define mm_is_pattern(typecode)	((typecode)[2]=='P')
#define mm_is_integer(typecode) ((typecode)[2]=='I')

#define mm_is_symmetric(typecode)((typecode)[3]=='S')
#define mm_is_general(typecode)	((typecode)[3]=='G')
#define mm_is_skew(typecode)	((typecode)[3]=='K')
#define mm_is_hermitian(typecode)((typecode)[3]=='H')

int mm_is_valid(MM_typecode matcode);		/* too complex for a macro */


/********************* MM_typecode modify fucntions ***************************/

#define mm_set_matrix(typecode)	((*typecode)[0]='M')
#define mm_set_coordinate(typecode)	((*typecode)[1]='C')
#define mm_set_array(typecode)	((*typecode)[1]='A')
#define mm_set_dense(typecode)	mm_set_array(typecode)
#define mm_set_sparse(typecode)	mm_set_coordinate(typecode)

#define mm_set_complex(typecode)((*typecode)[2]='C')
#define mm_set_real(typecode)	((*typecode)[2]='R')
#define mm_set_pattern(typecode)((*typecode)[2]='P')
#define mm_set_integer(typecode)((*typecode)[2]='I')


#define mm_set_symmetric(typecode)((*typecode)[3]='S')
#define mm_set_general(typecode)((*typecode)[3]='G')
#define mm_set_skew(typecode)	((*typecode)[3]='K')
#define mm_set_hermitian(typecode)((*typecode)[3]='H')

#define mm_clear_typecode(typecode) ((*typecode)[0]=(*typecode)[1]= \
                                    (*typecode)[2]=' ',(*typecode)[3]='G')

#define mm_initialize_typecode(typecode) mm_clear_typecode(typecode)


/********************* Matrix Market error codes ***************************/


#define MM_COULD_NOT_READ_FILE	11
#define MM_PREMATURE_EOF		12
#define MM_NOT_MTX				13
#define MM_NO_HEADER			14
#define MM_UNSUPPORTED_TYPE		15
#define MM_LINE_TOO_LONG		16
#define MM_COULD_NOT_WRITE_FILE	17


/******************** Matrix Market internal definitions ********************

   MM_matrix_typecode: 4-character sequence

                    ojbect 		sparse/   	data        storage
                                dense     	type        scheme

   string position:	 [0]        [1]			[2]         [3]

   Matrix typecode:  M(atrix)  C(oord)		R(eal)   	G(eneral)
                                A(array)	C(omplex)   H(ermitian)
                                            P(attern)   S(ymmetric)
                                            I(nteger)	K(kew)

 ***********************************************************************/

#define MM_MTX_STR		"matrix"
#define MM_ARRAY_STR	"array"
#define MM_DENSE_STR	"array"
#define MM_COORDINATE_STR "coordinate"
#define MM_SPARSE_STR	"coordinate"
#define MM_COMPLEX_STR	"complex"
#define MM_REAL_STR		"real"
#define MM_INT_STR		"integer"
#define MM_GENERAL_STR  "general"
#define MM_SYMM_STR		"symmetric"
#define MM_HERM_STR		"hermitian"
#define MM_SKEW_STR		"skew-symmetric"
#define MM_PATTERN_STR  "pattern"


/*  high level routines */

int mm_write_mtx_crd(char fname[], int M, int N, int nz, int I[], int J[],
         double val[], MM_typecode matcode);
int mm_read_mtx_crd_data(FILE *f, int M, int N, int nz, int I[], int J[],
        double val[], MM_typecode matcode);
int mm_read_mtx_crd_entry(FILE *f, int *I, int *J, double *real, double *img,
            MM_typecode matcode);

int mm_read_unsymmetric_sparse(const char *fname, int *M_, int *N_, int *nz_,
                double **val_, int **I_, int **J_);

char *mm_strdup(const char *s)
{
    int len = strlen(s);
    char *s2 = (char *) malloc((len+1)*sizeof(char));
    return strcpy(s2, s);
}

char  *mm_typecode_to_str(MM_typecode matcode)
{
    char buffer[MM_MAX_LINE_LENGTH];
    char *types[4];
    char *mm_strdup(const char *);
    //int error =0;

    /* check for MTX type */
    if (mm_is_matrix(matcode))
        types[0] = MM_MTX_STR;
    //else
    //    error=1;

    /* check for CRD or ARR matrix */
    if (mm_is_sparse(matcode))
        types[1] = MM_SPARSE_STR;
    else
    if (mm_is_dense(matcode))
        types[1] = MM_DENSE_STR;
    else
        return NULL;

    /* check for element data type */
    if (mm_is_real(matcode))
        types[2] = MM_REAL_STR;
    else
    if (mm_is_complex(matcode))
        types[2] = MM_COMPLEX_STR;
    else
    if (mm_is_pattern(matcode))
        types[2] = MM_PATTERN_STR;
    else
    if (mm_is_integer(matcode))
        types[2] = MM_INT_STR;
    else
        return NULL;


    /* check for symmetry type */
    if (mm_is_general(matcode))
        types[3] = MM_GENERAL_STR;
    else
    if (mm_is_symmetric(matcode))
        types[3] = MM_SYMM_STR;
    else
    if (mm_is_hermitian(matcode))
        types[3] = MM_HERM_STR;
    else
    if (mm_is_skew(matcode))
        types[3] = MM_SKEW_STR;
    else
        return NULL;

    sprintf(buffer,"%s %s %s %s", types[0], types[1], types[2], types[3]);
    return mm_strdup(buffer);

}

int mm_read_mtx_crd(char *fname, int *M, int *N, int *nz, int **I, int **J,
        double **val, MM_typecode *matcode)
{
    int ret_code;
    FILE *f;

    if (strcmp(fname, "stdin") == 0) f=stdin;
    else
    if ((f = fopen(fname, "r")) == NULL)
        return MM_COULD_NOT_READ_FILE;


    if ((ret_code = mm_read_banner(f, matcode)) != 0)
        return ret_code;

    if (!(mm_is_valid(*matcode) && mm_is_sparse(*matcode) &&
            mm_is_matrix(*matcode)))
        return MM_UNSUPPORTED_TYPE;

    if ((ret_code = mm_read_mtx_crd_size(f, M, N, nz)) != 0)
        return ret_code;


    *I = (int *)  malloc(*nz * sizeof(int));
    *J = (int *)  malloc(*nz * sizeof(int));
    *val = NULL;

    if (mm_is_complex(*matcode))
    {
        *val = (double *) malloc(*nz * 2 * sizeof(double));
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val,
                *matcode);
        if (ret_code != 0) return ret_code;
    }
    else if (mm_is_real(*matcode))
    {
        *val = (double *) malloc(*nz * sizeof(double));
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val,
                *matcode);
        if (ret_code != 0) return ret_code;
    }

    else if (mm_is_pattern(*matcode))
    {
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val,
                *matcode);
        if (ret_code != 0) return ret_code;
    }

    if (f != stdin) fclose(f);
    return 0;
}

int mm_read_banner(FILE *f, MM_typecode *matcode)
{
    char line[MM_MAX_LINE_LENGTH];
    char banner[MM_MAX_TOKEN_LENGTH];
    char mtx[MM_MAX_TOKEN_LENGTH];
    char crd[MM_MAX_TOKEN_LENGTH];
    char data_type[MM_MAX_TOKEN_LENGTH];
    char storage_scheme[MM_MAX_TOKEN_LENGTH];
    char *p;


    mm_clear_typecode(matcode);

    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
        return MM_PREMATURE_EOF;

    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type,
        storage_scheme) != 5)
        return MM_PREMATURE_EOF;

    for (p=mtx; *p!='\0'; *p=tolower(*p),p++);  /* convert to lower case */
    for (p=crd; *p!='\0'; *p=tolower(*p),p++);
    for (p=data_type; *p!='\0'; *p=tolower(*p),p++);
    for (p=storage_scheme; *p!='\0'; *p=tolower(*p),p++);

    /* check for banner */
    if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
        return MM_NO_HEADER;

    /* first field should be "mtx" */
    if (strcmp(mtx, MM_MTX_STR) != 0)
        return  MM_UNSUPPORTED_TYPE;
    mm_set_matrix(matcode);


    /* second field describes whether this is a sparse matrix (in coordinate
            storgae) or a dense array */


    if (strcmp(crd, MM_SPARSE_STR) == 0)
        mm_set_sparse(matcode);
    else
    if (strcmp(crd, MM_DENSE_STR) == 0)
            mm_set_dense(matcode);
    else
        return MM_UNSUPPORTED_TYPE;


    /* third field */

    if (strcmp(data_type, MM_REAL_STR) == 0)
        mm_set_real(matcode);
    else
    if (strcmp(data_type, MM_COMPLEX_STR) == 0)
        mm_set_complex(matcode);
    else
    if (strcmp(data_type, MM_PATTERN_STR) == 0)
        mm_set_pattern(matcode);
    else
    if (strcmp(data_type, MM_INT_STR) == 0)
        mm_set_integer(matcode);
    else
        return MM_UNSUPPORTED_TYPE;


    /* fourth field */

    if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
        mm_set_general(matcode);
    else
    if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
        mm_set_symmetric(matcode);
    else
    if (strcmp(storage_scheme, MM_HERM_STR) == 0)
        mm_set_hermitian(matcode);
    else
    if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
        mm_set_skew(matcode);
    else
        return MM_UNSUPPORTED_TYPE;


    return 0;
}

int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz)
{
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;

    /* set return null parameter values, in case we exit with errors */
    *M = *N = *nz = 0;

    /* now continue scanning until you reach the end-of-comments */
    do
    {
        if (fgets(line,MM_MAX_LINE_LENGTH,f) == NULL)
            return MM_PREMATURE_EOF;
    }while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d %d", M, N, nz) == 3)
        return 0;

    else
    do
    {
        num_items_read = fscanf(f, "%d %d %d", M, N, nz);
        if (num_items_read == EOF) return MM_PREMATURE_EOF;
    }
    while (num_items_read != 3);

    return 0;
}

int mm_read_mtx_array_size(FILE *f, int *M, int *N)
{
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;
    /* set return null parameter values, in case we exit with errors */
    *M = *N = 0;

    /* now continue scanning until you reach the end-of-comments */
    do
    {
        if (fgets(line,MM_MAX_LINE_LENGTH,f) == NULL)
            return MM_PREMATURE_EOF;
    }while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d", M, N) == 2)
        return 0;

    else /* we have a blank line */
    do
    {
        num_items_read = fscanf(f, "%d %d", M, N);
        if (num_items_read == EOF) return MM_PREMATURE_EOF;
    }
    while (num_items_read != 2);

    return 0;
}

int mm_write_banner(FILE *f, MM_typecode matcode)
{
    char *str = mm_typecode_to_str(matcode);
    int ret_code;

    ret_code = fprintf(f, "%s %s\n", MatrixMarketBanner, str);
    free(str);
    if (ret_code !=2 )
        return MM_COULD_NOT_WRITE_FILE;
    else
        return 0;
}

int mm_write_mtx_crd_size(FILE *f, int M, int N, int nz)
{
    if (fprintf(f, "%d %d %d\n", M, N, nz) != 3)
        return MM_COULD_NOT_WRITE_FILE;
    else
        return 0;
}

int mm_write_mtx_array_size(FILE *f, int M, int N)
{
    if (fprintf(f, "%d %d\n", M, N) != 2)
        return MM_COULD_NOT_WRITE_FILE;
    else
        return 0;
}




int mm_is_valid(MM_typecode matcode)		/* too complex for a macro */
{
    if (!mm_is_matrix(matcode)) return 0;
    if (mm_is_dense(matcode) && mm_is_pattern(matcode)) return 0;
    if (mm_is_real(matcode) && mm_is_hermitian(matcode)) return 0;
    if (mm_is_pattern(matcode) && (mm_is_hermitian(matcode) ||
                mm_is_skew(matcode))) return 0;
    return 1;
}




/*  high level routines */

int mm_write_mtx_crd(char fname[], int M, int N, int nz, int I[], int J[],
         double val[], MM_typecode matcode)
{
    FILE *f;
    int i;

    if (strcmp(fname, "stdout") == 0)
        f = stdout;
    else
    if ((f = fopen(fname, "w")) == NULL)
        return MM_COULD_NOT_WRITE_FILE;

    /* print banner followed by typecode */
    fprintf(f, "%s ", MatrixMarketBanner);
    fprintf(f, "%s\n", mm_typecode_to_str(matcode));

    /* print matrix sizes and nonzeros */
    fprintf(f, "%d %d %d\n", M, N, nz);

    /* print values */
    if (mm_is_pattern(matcode))
        for (i=0; i<nz; i++)
            fprintf(f, "%d %d\n", I[i], J[i]);
    else
    if (mm_is_real(matcode))
        for (i=0; i<nz; i++)
            fprintf(f, "%d %d %20.16g\n", I[i], J[i], val[i]);
    else
    if (mm_is_complex(matcode))
        for (i=0; i<nz; i++)
            fprintf(f, "%d %d %20.16g %20.16g\n", I[i], J[i], val[2*i],
                        val[2*i+1]);
    else
    {
        if (f != stdout) fclose(f);
        return MM_UNSUPPORTED_TYPE;
    }

    if (f !=stdout) fclose(f);

    return 0;
}

int mm_read_mtx_crd_data(FILE *f, int M, int N, int nz, int I[], int J[],
        double val[], MM_typecode matcode)
{
    int i;
    if (mm_is_complex(matcode))
    {
        for (i=0; i<nz; i++)
            if (fscanf(f, "%d %d %lg %lg", &I[i], &J[i], &val[2*i], &val[2*i+1])
                != 4) return MM_PREMATURE_EOF;
    }
    else if (mm_is_real(matcode))
    {
        for (i=0; i<nz; i++)
        {
            if (fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i])
                != 3) return MM_PREMATURE_EOF;

        }
    }

    else if (mm_is_pattern(matcode))
    {
        for (i=0; i<nz; i++)
            if (fscanf(f, "%d %d", &I[i], &J[i])
                != 2) return MM_PREMATURE_EOF;
    }
    else
        return MM_UNSUPPORTED_TYPE;

    return 0;

}

int mm_read_mtx_crd_entry(FILE *f, int *I, int *J, double *real, double *imag,
            MM_typecode matcode)
{
    if (mm_is_complex(matcode))
    {
            if (fscanf(f, "%d %d %lg %lg", I, J, real, imag)
                != 4) return MM_PREMATURE_EOF;
    }
    else if (mm_is_real(matcode))
    {
            if (fscanf(f, "%d %d %lg\n", I, J, real)
                != 3) return MM_PREMATURE_EOF;

    }

    else if (mm_is_pattern(matcode))
    {
            if (fscanf(f, "%d %d", I, J) != 2) return MM_PREMATURE_EOF;
    }
    else
        return MM_UNSUPPORTED_TYPE;

    return 0;

}

int mm_read_unsymmetric_sparse(const char *fname, int *M_, int *N_, int *nz_,
                double **val_, int **I_, int **J_)
{
    FILE *f;
    MM_typecode matcode;
    int M, N, nz;
    int i;
    double *val;
    int *I, *J;

    if ((f = fopen(fname, "r")) == NULL)
            return -1;


    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("mm_read_unsymetric: Could not process Matrix Market banner ");
        printf(" in file [%s]\n", fname);
        return -1;
    }



    if ( !(mm_is_real(matcode) && mm_is_matrix(matcode) &&
            mm_is_sparse(matcode)))
    {
        fprintf(stderr, "Sorry, this application does not support ");
        fprintf(stderr, "Market Market type: [%s]\n",
                mm_typecode_to_str(matcode));
        return -1;
    }

    /* find out size of sparse matrix: M, N, nz .... */

    if (mm_read_mtx_crd_size(f, &M, &N, &nz) !=0)
    {
        fprintf(stderr, "read_unsymmetric_sparse(): could not parse matrix size.\n");
        return -1;
    }

    *M_ = M;
    *N_ = N;
    *nz_ = nz;

    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));

    *val_ = val;
    *I_ = I;
    *J_ = J;

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }
    fclose(f);

    return 0;
}

enum data_type
{
    CHAR,
    UNSIGNED_CHAR,
    SHORT,
    UNSIGNED_SHORT,
    INT,
    UNSIGNED_INT,
    LONG,
    UNSIGNED_LONG,
    LONG_LONG,
    UNSIGNED_LONG_LONG,
    FLOAT,
    DOUBLE,
    BOOL
};

unsigned long read_from_array_with_data_type(void *arr, data_type type, unsigned long read_pos)
{
    assert(type == UNSIGNED_LONG || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_CHAR || type == BOOL);

    if (type == UNSIGNED_LONG)
    {
        unsigned long *output_arr = (unsigned long *)arr;
        return (unsigned long)(output_arr[read_pos]);
    }

    if (type == UNSIGNED_INT)
    {
        unsigned int *output_arr = (unsigned int *)arr;
        return (unsigned long)(output_arr[read_pos]);
    }

    if (type == UNSIGNED_SHORT)
    {
        unsigned short *output_arr = (unsigned short *)arr;
        return (unsigned short)(output_arr[read_pos]);
    }

    if (type == UNSIGNED_CHAR)
    {
        unsigned char *output_arr = (unsigned char *)arr;
        return (unsigned char)(output_arr[read_pos]);
    }

    if (type == BOOL)
    {
        bool *output_arr = (bool *)arr;
        return (bool)(output_arr[read_pos]);
    }

    cout << "error" << endl;
    exit(-1);
    return 0;
}

double read_double_from_array_with_data_type(void *arr, data_type type, unsigned long read_pos)
{
    assert(type == DOUBLE || type == FLOAT);

    if (type == DOUBLE)
    {
        double *output_arr = (double *)arr;
        return (double)(output_arr[read_pos]);
    }

    if (type == FLOAT)
    {
        float *output_arr = (float *)arr;
        return (double)(output_arr[read_pos]);
    }

    return 0;
}

void print_arr_to_file_with_data_type(void *arr, data_type type, unsigned long length, string file_name)
{
    assert(type == UNSIGNED_CHAR || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_LONG || type == DOUBLE || type == FLOAT || type == BOOL);
    ofstream arrWrite(file_name, ios::out | ios::trunc);

    if (type == UNSIGNED_CHAR || type == UNSIGNED_INT || type == UNSIGNED_SHORT || type == UNSIGNED_LONG || type == BOOL)
    {
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            arrWrite << read_from_array_with_data_type(arr, type, i) << endl;
        }
    }
    else if (type == DOUBLE || type == FLOAT)
    {
        unsigned long i;
        for (i = 0; i < length; i++)
        {
            arrWrite << read_double_from_array_with_data_type(arr, type, i) << endl;
        }
    }

    arrWrite.close();
}


void sort(int *col_idx, VALUE_TYPE *a, int start, int end)
{
	int i, j;
	int it;
	VALUE_TYPE dt;

	// 将每一行的列冒泡排序，保证每一行的列索引自增，时间特别长
	for (i = end - 1; i > start; i--)
	{
		for (j = start; j < i; j++)
		{
			if (col_idx[j] > col_idx[j + 1])
			{
				if (a)
				{
					dt = a[j];
					a[j] = a[j + 1];
					a[j + 1] = dt;
				}
				it = col_idx[j];
				col_idx[j] = col_idx[j + 1];
				col_idx[j + 1] = it;
			}
		}
	}
}

void coo2csr(int row_length, int nnz, VALUE_TYPE *values, int *row, int *col,
			 VALUE_TYPE *csr_values, int *col_idx, int *row_start)
{
	int i, l;
    assert(row != NULL);
    
	
	cout << "所有行全部初始化为0" << endl;
	for (i = 0; i <= row_length; i++)
	{
		row_start[i] = 0;
	}

    cout << "determine row lengths" << endl;
	for (i = 0; i < nnz; i++)
	{
		// 记录每一行的长度
        if (row[i] + 1 >= row_length + 1)
        {
            cout << "row[i] + 1: " << row[i] + 1 << endl;
            cout << "row_length + 1: " << row_length + 1 << endl;
        }
        assert(row[i] + 1 < row_length + 1);

		row_start[row[i] + 1]++;
	}

    cout << "将每一行的长度向前累加，获得CSR索引" << endl;
	for (i = 0; i < row_length; i++)
	{
		// 将每一行的长度向前累加，获得CSR索引
		row_start[i + 1] += row_start[i];
	}

	/* go through the structure  once more. Fill in output matrix. */
    cout << "go through the structure  once more. Fill in output matrix" << endl;
	// 这里假设原来的数组是完全乱序的
	for (l = 0; l < nnz; l++)
	{
		i = row_start[row[l]];
		csr_values[i] = values[l];
		col_idx[i] = col[l];
		row_start[row[l]]++;
	}

    cout << "shift back row_start" << endl;
	/* shift back row_start */
	for (i = row_length; i > 0; i--)
	{
		row_start[i] = row_start[i - 1];
	}

	row_start[0] = 0;

	cout << "sort" << endl;
	// 遍历每一行
	// for (i = 0; i < row_length; i++)
	// {
	// 	// cout << i << " " << row_start[i] << " " << row_start[i + 1] << endl;
	// 	sort(col_idx, csr_values, row_start[i], row_start[i + 1]);
	// }

	cout << "finish sort" << endl;
}

unsigned long dummy_var = 0;

// 用两个全局变量存储所有的预处理时间
struct timeval pre_start, pre_end;

static void conv(string data_file_name, int* row_off, int* col_idx, VALUE_TYPE* values, int& nnz_max, bool isData = false, unsigned long &nnz_avg = dummy_var, unsigned long &nnz_dev = dummy_var)
{
	string d;
	d = data_file_name;
	std::ifstream fin(d.c_str());
	if (!fin)
	{
		cout << "File Not found\n";
		exit(0);
	}

	int row_length, column_length, nnz;

	// Ignore headers and comments:
	while (fin.peek() == '%')
		fin.ignore(2048, '\n');

	// Read defining parameters:
	fin >> row_length >> column_length >> nnz;

	// Create your matrix:
	int *row, *column;
	VALUE_TYPE *coovalues;
	row = new int[nnz];
	column = new int[nnz];
	coovalues = new VALUE_TYPE[nnz];
	
	std::fill(row, row + nnz, 0);
	std::fill(column, column + nnz, 0);
	std::fill(coovalues, coovalues + nnz, 0);

	// Read the data
	for (int l = 0; l < nnz; l++)
	{
		int m, n;
		VALUE_TYPE data;
		if (!isData)
			fin >> m >> n; // >> data;
		else
			fin >> m >> n >> data;
        
		row[l] = m - 1;
        
        if (!(m - 1 >= 0 && m - 1 < row_length))
        {
            cout << "m - 1:" << m - 1 << ", l:" << l << endl;
        }
        
        assert(m - 1 >= 0 && m - 1 < row_length);
		column[l] = n - 1;

        if (!(n - 1 >= 0 && n - 1 < column_length))
        {
            cout << "n - 1:" << n - 1 << ", l:" << l << endl;
        }

        assert(n - 1 >= 0 && n - 1 < column_length);
		if (!isData)
			coovalues[l] = 1;
		else
			coovalues[l] = data;
	}

    // 这里是预处理的开始
    gettimeofday(&pre_start, NULL);
    
	cout << "get coo data" << endl;

	coo2csr(row_length, nnz, coovalues, row, column, values, col_idx, row_off);

	cout << "get csr data" << endl;

	nnz_max = 0;

	int tot_nnz = 0, tot_nnz_square = 0;
	for (int i = 0; i < row_length - 1; i++)
	{
		int cur_nnz = row_off[i + 1] - row_off[i];
		tot_nnz += cur_nnz;
		tot_nnz_square += cur_nnz * cur_nnz;
		if (cur_nnz > nnz_max)
		{
			nnz_max = cur_nnz;
			// cout << "nnz_max_row:" << i << endl;
		}

	}

	tot_nnz += nnz - row_off[row_length - 1];
	tot_nnz_square += (nnz - row_off[row_length - 1]);

	if ((nnz - row_off[row_length - 1]) > nnz_max)
	{
		nnz_max = nnz - row_off[row_length - 1];
	}

	nnz_avg = tot_nnz / row_length;
	nnz_dev = (int)sqrt(tot_nnz_square / row_length - (nnz_avg * nnz_avg));

	row_off[row_length] = nnz;

	delete[] row;
	delete[] column;
	delete[] coovalues;

	fin.close();
	
	cout << "begin div block" << endl;
}



#endif
