/**
 * Linear algebra builtin functions
 */

import { registerNorm } from "./norm.js";
import { registerDot } from "./dot.js";
import { registerQr } from "./qr.js";
import { registerInv } from "./inv.js";
import { registerDet } from "./det.js";
import { registerSvd } from "./svd.js";
import { registerLinsolve } from "./linsolve.js";
import { registerFft } from "./fft.js";
import { registerKron } from "./kron.js";
import { registerEig } from "./eig.js";
import { registerCond } from "./cond.js";
import { registerRank } from "./rank.js";
import { registerPagemtimes } from "./pagemtimes.js";
import { registerPagetranspose } from "./pagetranspose.js";
import { registerLu } from "./lu.js";
import { registerBlkdiag } from "./blkdiag.js";
import { registerChol } from "./chol.js";
import { registerPinv } from "./pinv.js";
import { registerQz } from "./qz.js";
import { registerVecnorm } from "./vecnorm.js";

export function registerLinearAlgebraFunctions(): void {
  registerNorm();
  registerDot();
  registerQr();
  registerInv();
  registerDet();
  registerSvd();
  registerLinsolve();
  registerFft();
  registerKron();
  registerEig();
  registerCond();
  registerRank();
  registerPagemtimes();
  registerPagetranspose();
  registerLu();
  registerBlkdiag();
  registerChol();
  registerPinv();
  registerQz();
  registerVecnorm();
}
