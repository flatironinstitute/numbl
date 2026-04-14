/**
 * numbl_ops — shared bits: error strings, op-code dump.
 */

#include "numbl_ops.h"

#include <stdio.h>
#include <string.h>

const char* numbl_strerror(int code) {
  switch (code) {
    case NUMBL_OK:           return "ok";
    case NUMBL_ERR_BAD_OP:   return "unknown op code";
    case NUMBL_ERR_NULL_PTR: return "null pointer argument";
    default:                 return "unknown numbl error";
  }
}

/* Dump op-code enum values in a stable text format.
 * Used by the TS-side drift-detection test.
 */
size_t numbl_dump_op_codes(char* buf, size_t buf_size) {
  /* Compose into a temporary; emit length only if buf is too small. */
  char tmp[1024];
  int n = 0;
  n += snprintf(tmp + n, sizeof(tmp) - n,
                "real_binary:ADD=%d,SUB=%d,MUL=%d,DIV=%d;",
                NUMBL_REAL_BIN_ADD, NUMBL_REAL_BIN_SUB,
                NUMBL_REAL_BIN_MUL, NUMBL_REAL_BIN_DIV);
  n += snprintf(tmp + n, sizeof(tmp) - n,
                "complex_binary:ADD=%d,SUB=%d,MUL=%d,DIV=%d;",
                NUMBL_COMPLEX_BIN_ADD, NUMBL_COMPLEX_BIN_SUB,
                NUMBL_COMPLEX_BIN_MUL, NUMBL_COMPLEX_BIN_DIV);
  n += snprintf(tmp + n, sizeof(tmp) - n,
                "unary:EXP=%d,LOG=%d,LOG2=%d,LOG10=%d,SQRT=%d,ABS=%d,"
                "FLOOR=%d,CEIL=%d,ROUND=%d,TRUNC=%d,"
                "SIN=%d,COS=%d,TAN=%d,ASIN=%d,ACOS=%d,ATAN=%d,"
                "SINH=%d,COSH=%d,TANH=%d,SIGN=%d;",
                NUMBL_UNARY_EXP, NUMBL_UNARY_LOG, NUMBL_UNARY_LOG2,
                NUMBL_UNARY_LOG10, NUMBL_UNARY_SQRT, NUMBL_UNARY_ABS,
                NUMBL_UNARY_FLOOR, NUMBL_UNARY_CEIL, NUMBL_UNARY_ROUND,
                NUMBL_UNARY_TRUNC, NUMBL_UNARY_SIN, NUMBL_UNARY_COS,
                NUMBL_UNARY_TAN, NUMBL_UNARY_ASIN, NUMBL_UNARY_ACOS,
                NUMBL_UNARY_ATAN, NUMBL_UNARY_SINH, NUMBL_UNARY_COSH,
                NUMBL_UNARY_TANH, NUMBL_UNARY_SIGN);
  n += snprintf(tmp + n, sizeof(tmp) - n,
                "cmp:EQ=%d,NE=%d,LT=%d,LE=%d,GT=%d,GE=%d;",
                NUMBL_CMP_EQ, NUMBL_CMP_NE, NUMBL_CMP_LT,
                NUMBL_CMP_LE, NUMBL_CMP_GT, NUMBL_CMP_GE);
  n += snprintf(tmp + n, sizeof(tmp) - n,
                "reduce:SUM=%d,PROD=%d,MAX=%d,MIN=%d,ANY=%d,ALL=%d,MEAN=%d;",
                NUMBL_REDUCE_SUM, NUMBL_REDUCE_PROD, NUMBL_REDUCE_MAX,
                NUMBL_REDUCE_MIN, NUMBL_REDUCE_ANY, NUMBL_REDUCE_ALL,
                NUMBL_REDUCE_MEAN);
  size_t need = (size_t)n;
  if (buf && buf_size > need) {
    memcpy(buf, tmp, need + 1);
  } else if (buf && buf_size > 0) {
    buf[0] = '\0';
  }
  return need;
}
