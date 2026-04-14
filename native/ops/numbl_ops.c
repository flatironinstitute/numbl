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
  size_t need = (size_t)n;
  if (buf && buf_size > need) {
    memcpy(buf, tmp, need + 1);
  } else if (buf && buf_size > 0) {
    buf[0] = '\0';
  }
  return need;
}
