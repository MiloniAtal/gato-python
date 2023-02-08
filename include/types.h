#ifndef GATO_TYPES_H
#define GATO_TYPES_H


/*   CSR format matrix   */
typedef struct _csr_t {
  unsigned               m;          ///< number of rows
  unsigned               n;          ///< number of columns
  unsigned               *row_ptr;    ///< row pointers (size m+1)
  unsigned               *col_ind;    ///< column indices (size nnz)
  float                  *val;        ///< numerical values (size nnz)
  unsigned               nnz;        ///< number of non-zero entries in matrix
} csr_t;

/*   vector   */
typedef struct _vec_t {
    float       *val;
    unsigned    len;
} vec_t;


typedef struct _kkt_t {
    csr_t *G;
    csr_t *C;
    vec_t *g;
    vec_t *c;

    vec_t *dz;
    vec_t *lambda;
} kkt_t;

#endif /* GATO_TYPES_H */