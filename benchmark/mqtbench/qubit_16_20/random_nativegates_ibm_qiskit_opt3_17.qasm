// Benchmark was created by MQT Bench on 2024-03-18
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 1.1.0
// Qiskit version: 1.0.2
// Used Gate Set: ['id', 'rz', 'sx', 'x', 'cx', 'measure', 'barrier']

OPENQASM 2.0;
include "qelib1.inc";
qreg q[17];
creg meas[17];
rz(-3.0265869816947912) q[1];
rz(-3.026586981694792) q[2];
rz(-pi) q[3];
sx q[3];
rz(2.0790579125975137) q[3];
sx q[3];
cx q[0],q[3];
sx q[3];
rz(2.0790579125975137) q[3];
sx q[3];
rz(-pi) q[3];
cx q[0],q[3];
rz(-3.305156200563297) q[0];
sx q[0];
rz(8.404160964631544) q[0];
sx q[0];
rz(12.729934161332675) q[0];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
cx q[1],q[4];
x q[1];
rz(pi/4) q[4];
cx q[1],q[4];
x q[1];
rz(2.241188818297342) q[1];
rz(3*pi/4) q[4];
sx q[4];
rz(-0.42973578711346505) q[4];
sx q[4];
sx q[6];
rz(1.5915425546174937) q[6];
sx q[6];
rz(-pi) q[6];
rz(pi/2) q[7];
cx q[2],q[7];
x q[2];
rz(pi/4) q[7];
cx q[2],q[7];
rz(-0.036287502898395996) q[2];
sx q[2];
rz(-1.5640163777242586) q[2];
sx q[2];
rz(-2.7112359532258683) q[2];
rz(-0.15340334460671112) q[7];
sx q[7];
rz(-3.0663008923806228) q[7];
sx q[7];
rz(-3*pi/2) q[7];
rz(-3.615635604242877) q[8];
cx q[8],q[3];
rz(-pi/16) q[3];
cx q[8],q[3];
rz(pi/16) q[3];
cx q[8],q[6];
rz(-pi/16) q[6];
cx q[6],q[3];
rz(pi/16) q[3];
cx q[6],q[3];
rz(-pi/16) q[3];
cx q[8],q[6];
rz(pi/16) q[6];
cx q[6],q[3];
rz(-pi/16) q[3];
cx q[6],q[3];
rz(pi/16) q[3];
cx q[6],q[1];
rz(-pi/16) q[1];
cx q[1],q[3];
rz(pi/16) q[3];
cx q[1],q[3];
rz(-pi/16) q[3];
cx q[8],q[1];
rz(pi/16) q[1];
cx q[1],q[3];
rz(-pi/16) q[3];
cx q[1],q[3];
rz(pi/16) q[3];
cx q[6],q[1];
cx q[0],q[6];
rz(-pi/16) q[1];
cx q[1],q[3];
rz(pi/16) q[3];
cx q[1],q[3];
rz(-pi/16) q[3];
cx q[6],q[0];
cx q[0],q[6];
sx q[0];
rz(3.5831736447277134) q[0];
cx q[8],q[1];
rz(-4.516039439535327) q[1];
cx q[1],q[3];
rz(-pi/16) q[3];
cx q[1],q[3];
sx q[1];
rz(3*pi/4) q[1];
rz(9*pi/16) q[3];
sx q[3];
rz(2.071000093779598) q[3];
rz(-pi/2) q[9];
sx q[9];
rz(-2.0468138885037392) q[9];
sx q[9];
rz(1.6858019986898984) q[9];
rz(-3*pi/2) q[10];
sx q[10];
rz(3*pi/4) q[10];
cx q[5],q[10];
rz(-pi/4) q[10];
rz(pi/2) q[11];
rz(0.943821156561496) q[12];
sx q[12];
rz(-pi/2) q[12];
sx q[13];
rz(pi/2) q[13];
cx q[2],q[13];
rz(-pi/4) q[13];
cx q[2],q[13];
sx q[2];
rz(pi/4) q[13];
cx q[14],q[10];
rz(pi/4) q[10];
cx q[5],q[10];
rz(-pi) q[5];
sx q[5];
rz(0.115005671895001) q[5];
rz(pi/4) q[10];
sx q[10];
rz(-0.12997299123186368) q[10];
rz(-1.1003057807775) q[15];
sx q[15];
cx q[12],q[15];
sx q[12];
rz(-0.1353209511116562) q[12];
sx q[12];
rz(0.1353209511116577) q[15];
cx q[12],q[15];
rz(-pi/2) q[12];
sx q[12];
rz(-2.5146174833563926) q[12];
cx q[10],q[12];
x q[10];
rz(pi/4) q[12];
cx q[10],q[12];
x q[10];
rz(-1.4078742709827319) q[10];
cx q[5],q[10];
x q[5];
rz(pi/4) q[10];
cx q[5],q[10];
x q[5];
rz(-2.4712001620873476) q[5];
x q[10];
rz(3.989326587650056) q[10];
x q[12];
rz(-3*pi/4) q[12];
cx q[12],q[1];
rz(pi/4) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-1.7646793129738874) q[15];
sx q[15];
rz(-1.9329274604762823) q[15];
sx q[15];
rz(-2.6354999879656766) q[15];
cx q[14],q[15];
sx q[15];
rz(0.40873928077988664) q[15];
sx q[15];
rz(-pi) q[15];
cx q[14],q[15];
rz(-2.068898827008697) q[14];
sx q[14];
rz(-2.644371446677079) q[14];
sx q[14];
rz(-2.4488726882803515) q[14];
cx q[10],q[14];
sx q[14];
rz(1.175198342799403) q[14];
sx q[14];
rz(-pi) q[14];
cx q[10],q[14];
rz(-pi) q[14];
sx q[14];
rz(-pi) q[14];
cx q[15],q[1];
rz(pi/4) q[1];
cx q[3],q[1];
rz(-pi/4) q[1];
cx q[15],q[1];
rz(pi/4) q[1];
cx q[3],q[1];
rz(pi/4) q[1];
sx q[1];
rz(3*pi/4) q[1];
cx q[12],q[1];
rz(-3*pi/4) q[1];
sx q[1];
rz(6.004767939320555) q[1];
sx q[1];
rz(9.686287903574188) q[1];
sx q[12];
rz(pi/2) q[12];
cx q[0],q[12];
rz(3.981707520827624) q[12];
cx q[0],q[12];
sx q[0];
rz(-1.3220750844461708) q[0];
sx q[0];
rz(-3.1282252148179204) q[0];
rz(pi/2) q[12];
sx q[12];
rz(-pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
rz(2.6819307806757893) q[15];
cx q[15],q[13];
rz(-1.111134453880893) q[13];
cx q[15],q[13];
rz(-2.0304581997089004) q[13];
sx q[13];
rz(3*pi/4) q[13];
sx q[15];
rz(8.191704657025124) q[15];
sx q[15];
rz(5*pi/2) q[15];
sx q[16];
cx q[11],q[16];
x q[11];
cx q[11],q[4];
sx q[4];
rz(2.7118568664763263) q[4];
sx q[4];
rz(-pi) q[4];
cx q[11],q[4];
rz(-0.2959258889794909) q[4];
sx q[4];
rz(4.982524896786419) q[4];
sx q[4];
rz(7.369125071965419) q[4];
rz(2.148184855860128) q[11];
sx q[11];
rz(-pi) q[11];
sx q[16];
rz(-pi) q[16];
cx q[9],q[16];
x q[9];
rz(0.630629614676606) q[16];
cx q[9],q[16];
rz(0.11500567189499966) q[9];
sx q[9];
rz(-1.4247840690836195) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[11];
sx q[9];
rz(-0.4253617599041233) q[9];
sx q[9];
rz(0.4253617599041247) q[11];
cx q[9],q[11];
rz(pi/2) q[9];
sx q[9];
rz(-0.18041858865023475) q[9];
cx q[5],q[9];
rz(-1.5363899958559373) q[9];
cx q[5],q[9];
sx q[5];
rz(3.163607330757486) q[5];
rz(-pi/2) q[9];
sx q[9];
rz(3*pi/4) q[9];
sx q[11];
rz(2.14818485586013) q[11];
cx q[7],q[11];
cx q[7],q[15];
cx q[11],q[14];
cx q[14],q[11];
rz(pi) q[14];
x q[14];
rz(5.937976762063549) q[15];
cx q[7],q[15];
sx q[7];
rz(pi/2) q[7];
rz(-2.134942074530997) q[15];
sx q[15];
rz(-1.4955685619292716) q[15];
sx q[15];
rz(0.07240606139166772) q[15];
sx q[16];
rz(0.25922242164187903) q[16];
cx q[8],q[16];
x q[8];
rz(pi/4) q[16];
cx q[8],q[16];
x q[8];
rz(-1.812844848415672) q[8];
cx q[6],q[8];
rz(-2.229151640466572) q[8];
cx q[6],q[8];
rz(pi/2) q[6];
cx q[6],q[1];
rz(-pi/4) q[1];
cx q[6],q[1];
rz(0.1458714339507301) q[1];
cx q[1],q[5];
rz(3.2440404314539117) q[5];
cx q[1],q[5];
sx q[1];
rz(-pi/2) q[1];
rz(pi/2) q[5];
sx q[5];
rz(3.1433479536006472) q[5];
cx q[6],q[12];
rz(-pi/2) q[8];
sx q[8];
rz(-pi/2) q[8];
cx q[8],q[2];
rz(pi/2) q[2];
sx q[8];
rz(-pi) q[8];
cx q[8],q[2];
rz(pi/4) q[2];
sx q[8];
cx q[8],q[2];
rz(2.3805452784418275) q[2];
sx q[2];
cx q[1],q[2];
sx q[1];
rz(-0.4268383783413481) q[1];
sx q[1];
rz(0.4268383783413496) q[2];
cx q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-2.502065924143076) q[1];
sx q[1];
rz(3*pi/4) q[1];
sx q[2];
rz(2.331843701942864) q[2];
rz(-1.16659103373261) q[8];
sx q[8];
rz(-2.466367483079817) q[8];
sx q[8];
rz(0.33988094895248544) q[8];
rz(-pi/4) q[12];
cx q[6],q[12];
sx q[6];
rz(0.4853568916020463) q[6];
sx q[6];
rz(0.4923892188215122) q[6];
rz(3*pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[14],q[1];
rz(pi/4) q[1];
sx q[1];
cx q[14],q[2];
rz(-pi/4) q[2];
cx q[1],q[2];
rz(pi/4) q[2];
cx q[14],q[2];
rz(-pi/4) q[2];
cx q[1],q[2];
rz(2.1547866895110834) q[2];
sx q[2];
rz(-1.2422612462072475) q[2];
sx q[2];
rz(0.2690384823279093) q[2];
rz(pi/4) q[14];
cx q[1],q[14];
rz(3.478262315256173) q[1];
rz(-pi/4) q[14];
cx q[1],q[14];
sx q[1];
rz(-2.0628768932880126) q[1];
sx q[1];
rz(2.3537811775399193) q[1];
rz(1.0446205850393255) q[16];
sx q[16];
rz(-0.3022275588416363) q[16];
sx q[16];
cx q[3],q[16];
sx q[16];
rz(2.839365094748156) q[16];
sx q[16];
rz(-pi) q[16];
cx q[3],q[16];
cx q[3],q[13];
rz(pi/4) q[13];
sx q[13];
cx q[13],q[9];
rz(pi/4) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[3],q[9];
rz(pi/4) q[9];
cx q[0],q[9];
rz(-pi/4) q[9];
cx q[3],q[9];
sx q[3];
rz(-2.198056257685149) q[3];
sx q[3];
rz(0.21985932769820948) q[3];
rz(pi/4) q[9];
cx q[0],q[9];
cx q[5],q[0];
rz(-pi/4) q[0];
cx q[5],q[0];
rz(-1.4056040885855279) q[0];
cx q[4],q[0];
rz(-0.4413406227301778) q[0];
cx q[4],q[0];
rz(pi/2) q[0];
sx q[0];
rz(-0.6392608794381722) q[0];
sx q[0];
rz(-pi/2) q[0];
cx q[0],q[1];
sx q[0];
rz(-pi/2) q[1];
sx q[1];
rz(-0.7380802994812505) q[1];
sx q[1];
rz(pi/4) q[1];
rz(pi/4) q[9];
sx q[9];
rz(3*pi/4) q[9];
cx q[13],q[9];
rz(pi/4) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[12];
cx q[12],q[9];
rz(-1.5690440902867862) q[9];
sx q[9];
rz(-1.3546554316776174) q[9];
sx q[9];
rz(0.21488792271358825) q[9];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[6],q[12];
rz(-pi/4) q[12];
cx q[6],q[12];
sx q[6];
rz(5.419831282588547) q[6];
sx q[6];
rz(6.338092852335912) q[6];
rz(3*pi/4) q[12];
sx q[12];
rz(0.43510419851743887) q[12];
sx q[13];
rz(-2.4035123541085435) q[13];
sx q[13];
rz(-pi/2) q[13];
sx q[16];
rz(-pi/2) q[16];
cx q[16],q[10];
sx q[10];
rz(2.261354539616276) q[10];
sx q[10];
rz(-pi) q[10];
sx q[16];
rz(2.261354539616276) q[16];
sx q[16];
cx q[16],q[10];
rz(-2.138921755524869) q[10];
cx q[10],q[13];
sx q[10];
rz(pi/4) q[10];
cx q[11],q[10];
rz(pi/4) q[10];
rz(pi/2) q[13];
sx q[13];
rz(-2.4035123541085426) q[13];
sx q[13];
cx q[13],q[10];
rz(-pi/4) q[10];
cx q[11],q[10];
rz(3*pi/4) q[10];
sx q[10];
rz(pi) q[10];
rz(pi/4) q[13];
cx q[11],q[13];
rz(2.651359428170859) q[11];
rz(-pi/4) q[13];
cx q[11],q[13];
cx q[10],q[13];
sx q[10];
rz(2.3137871193385298) q[10];
sx q[10];
cx q[11],q[8];
rz(0.9290828090748837) q[8];
sx q[8];
rz(-2.5540962620590575) q[8];
sx q[8];
cx q[11],q[8];
sx q[8];
rz(-2.5540962620590584) q[8];
sx q[8];
rz(0.3018325687675265) q[8];
cx q[11],q[2];
rz(-pi/4) q[2];
cx q[11],q[2];
rz(3*pi/4) q[2];
sx q[2];
rz(-0.09029974716917266) q[2];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[7];
rz(2.2523109408064848) q[7];
cx q[13],q[7];
rz(pi/2) q[7];
sx q[7];
rz(6.461526429898031) q[7];
cx q[7],q[10];
sx q[10];
rz(2.313787119338529) q[10];
sx q[10];
rz(-pi) q[10];
cx q[7],q[10];
cx q[7],q[4];
rz(-1.7491374495133414) q[4];
sx q[4];
rz(-2.431332609650048) q[4];
sx q[4];
cx q[7],q[4];
sx q[4];
rz(-2.431332609650048) q[4];
sx q[4];
rz(-0.6342141273082733) q[4];
cx q[4],q[1];
rz(pi/4) q[1];
cx q[0],q[1];
rz(pi/4) q[0];
rz(-pi/4) q[1];
cx q[4],q[1];
rz(3*pi/4) q[1];
sx q[1];
rz(3.0637630763961523) q[1];
cx q[4],q[0];
rz(-pi/4) q[0];
cx q[4],q[0];
cx q[1],q[0];
sx q[0];
rz(pi) q[0];
sx q[1];
rz(-1.5885285234875814) q[1];
sx q[1];
rz(-0.10855535876931555) q[1];
sx q[7];
sx q[13];
rz(-1.7565120451842642) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[7];
x q[13];
rz(0.07330247784041966) q[13];
cx q[14],q[8];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[12],q[14];
rz(-pi/4) q[14];
cx q[12],q[14];
cx q[12],q[7];
cx q[7],q[12];
cx q[12],q[7];
sx q[7];
rz(2.6427264757350093) q[7];
sx q[7];
rz(0.11500567189500543) q[7];
rz(pi/4) q[12];
sx q[12];
rz(-2.4035123541085435) q[12];
sx q[12];
rz(-pi/2) q[12];
rz(3*pi/4) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[4],q[14];
rz(-1.9189826948777495) q[14];
cx q[4],q[14];
sx q[4];
rz(3*pi/4) q[4];
rz(0.3481863680828532) q[14];
x q[16];
cx q[16],q[15];
rz(-pi/4) q[15];
sx q[15];
rz(pi/4) q[15];
cx q[10],q[15];
rz(-pi/4) q[15];
cx q[5],q[15];
sx q[5];
rz(-0.017024633722080296) q[5];
sx q[5];
rz(pi/4) q[5];
rz(pi/4) q[15];
cx q[10],q[15];
cx q[10],q[9];
rz(3*pi/4) q[9];
sx q[9];
rz(-pi) q[9];
cx q[8],q[9];
x q[8];
rz(-2.7297131381303945) q[8];
cx q[1],q[8];
x q[1];
rz(1.1600854394428957) q[8];
cx q[1],q[8];
rz(-1.1933432303839648) q[1];
sx q[1];
rz(-3.1238100801480337) q[1];
sx q[1];
rz(-1.6460820345785212) q[1];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/4) q[9];
cx q[9],q[4];
rz(-pi/4) q[4];
cx q[2],q[4];
rz(pi/4) q[4];
cx q[9],q[4];
rz(pi/4) q[4];
sx q[4];
rz(3*pi/4) q[4];
cx q[4],q[2];
rz(-pi/4) q[2];
cx q[4],q[2];
rz(3*pi/4) q[2];
sx q[2];
rz(pi/2) q[2];
sx q[9];
rz(1.4507329633265051) q[9];
cx q[10],q[5];
rz(pi/4) q[5];
sx q[5];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[11],q[10];
rz(-pi/16) q[10];
cx q[11],q[10];
rz(pi/16) q[10];
cx q[11],q[5];
rz(-pi/16) q[5];
cx q[5],q[10];
rz(pi/16) q[10];
cx q[5],q[10];
rz(-pi/16) q[10];
cx q[11],q[5];
rz(9*pi/16) q[5];
cx q[5],q[10];
rz(-pi/16) q[10];
cx q[5],q[10];
rz(pi/16) q[10];
rz(-pi/4) q[15];
sx q[15];
rz(-pi/4) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[3],q[15];
sx q[3];
rz(-0.5630186708707172) q[3];
sx q[3];
rz(pi/2) q[15];
cx q[3],q[15];
rz(-pi) q[3];
sx q[3];
rz(3.99697881713917) q[3];
cx q[3],q[8];
rz(3.943984945955521) q[8];
cx q[3],q[8];
sx q[3];
rz(-0.42273764678309345) q[8];
sx q[8];
rz(-2.3212528224917968) q[8];
sx q[8];
rz(3.1249261536397137) q[8];
cx q[1],q[8];
cx q[8],q[1];
cx q[1],q[8];
rz(-pi/2) q[8];
sx q[8];
rz(-pi/2) q[8];
sx q[15];
rz(2.561062051692309) q[15];
sx q[15];
cx q[15],q[14];
rz(pi/2) q[14];
rz(0.9377220734486329) q[16];
sx q[16];
rz(-2.482760446707715) q[16];
sx q[16];
rz(0.033544415939687156) q[16];
cx q[6],q[16];
sx q[6];
rz(pi/2) q[6];
cx q[0],q[6];
rz(0.2643182647519992) q[6];
cx q[0],q[6];
sx q[0];
rz(pi/2) q[6];
sx q[6];
rz(-3*pi/4) q[6];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[13];
sx q[13];
rz(1.3831395554700627) q[13];
sx q[13];
rz(-pi) q[13];
sx q[16];
rz(1.3831395554700627) q[16];
sx q[16];
rz(-5*pi/2) q[16];
cx q[16],q[13];
rz(1.497493848954477) q[13];
cx q[7],q[13];
x q[7];
rz(0.420012369514814) q[13];
cx q[7],q[13];
rz(-3.026586981694792) q[7];
sx q[7];
rz(-2.0324231699268394) q[7];
sx q[7];
rz(pi/16) q[7];
x q[13];
rz(-7*pi/16) q[13];
cx q[13],q[14];
rz(-pi/16) q[14];
cx q[14],q[7];
rz(pi/16) q[7];
cx q[14],q[7];
rz(-pi/16) q[7];
cx q[13],q[14];
rz(5*pi/16) q[14];
cx q[14],q[7];
rz(-pi/16) q[7];
cx q[14],q[7];
rz(pi/16) q[7];
cx q[14],q[15];
rz(-pi/16) q[15];
cx q[15],q[7];
rz(pi/16) q[7];
cx q[15],q[7];
rz(-pi/16) q[7];
cx q[13],q[15];
rz(pi/16) q[15];
cx q[15],q[7];
rz(-pi/16) q[7];
cx q[15],q[7];
rz(pi/16) q[7];
cx q[14],q[15];
rz(-pi/16) q[15];
cx q[15],q[7];
rz(pi/16) q[7];
cx q[15],q[7];
rz(-pi/16) q[7];
cx q[13],q[15];
sx q[13];
rz(0.507396472691648) q[13];
sx q[13];
cx q[3],q[13];
sx q[13];
rz(0.5073964726916471) q[13];
sx q[13];
rz(-pi) q[13];
cx q[3],q[13];
sx q[3];
rz(-3*pi/4) q[3];
rz(-pi/2) q[13];
rz(-2.6837923038575386) q[15];
cx q[15],q[7];
rz(-pi/16) q[7];
cx q[15],q[7];
rz(9*pi/16) q[7];
sx q[7];
rz(pi) q[7];
cx q[7],q[2];
rz(1.6790716531569359) q[2];
cx q[7],q[2];
rz(-2.9528141797233634) q[2];
sx q[2];
rz(-1.3852945778483132) q[2];
sx q[2];
rz(-2.338582859040603) q[2];
sx q[7];
rz(pi/2) q[7];
sx q[15];
rz(-0.6349702169407827) q[15];
sx q[15];
rz(-1.2499263669268448) q[15];
sx q[16];
rz(-pi/2) q[16];
cx q[5],q[16];
rz(-pi/16) q[16];
cx q[16],q[10];
rz(pi/16) q[10];
cx q[16],q[10];
rz(-pi/16) q[10];
cx q[11],q[16];
rz(pi/16) q[16];
cx q[16],q[10];
rz(-pi/16) q[10];
cx q[16],q[10];
cx q[5],q[16];
cx q[5],q[6];
sx q[5];
rz(5*pi/4) q[5];
sx q[5];
rz(3*pi/4) q[5];
rz(-pi/2) q[6];
rz(pi/16) q[10];
cx q[14],q[6];
rz(3.0008683716973206) q[6];
sx q[6];
rz(-1.7901750044722595) q[6];
sx q[6];
rz(-4.4275735729730865) q[6];
rz(-pi/16) q[16];
cx q[16],q[10];
rz(pi/16) q[10];
cx q[16],q[10];
rz(-pi/16) q[10];
cx q[11],q[16];
cx q[11],q[7];
rz(-pi/4) q[7];
cx q[11],q[7];
rz(3*pi/4) q[7];
sx q[7];
rz(-1.3562939513576993) q[7];
cx q[9],q[7];
sx q[7];
rz(2.3106531205266974) q[7];
sx q[7];
rz(-pi) q[7];
sx q[9];
rz(2.3106531205266974) q[9];
sx q[9];
cx q[9],q[7];
rz(-1.7852987022320943) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-3*pi/2) q[9];
sx q[9];
rz(-3*pi/2) q[9];
sx q[11];
rz(3.6137375882004394) q[11];
sx q[11];
rz(9.918226873088669) q[11];
rz(-2.6002198865864132) q[16];
cx q[16],q[10];
rz(-pi/16) q[10];
cx q[16],q[10];
rz(9*pi/16) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[12];
sx q[10];
rz(pi/4) q[10];
cx q[0],q[10];
rz(pi/4) q[10];
rz(pi/2) q[12];
sx q[12];
rz(-2.4035123541085426) q[12];
sx q[12];
cx q[12],q[10];
rz(-pi/4) q[10];
cx q[0],q[10];
rz(3*pi/4) q[10];
sx q[10];
rz(pi) q[10];
rz(pi/4) q[12];
cx q[0],q[12];
rz(-pi/4) q[0];
rz(-pi/4) q[12];
cx q[0],q[12];
sx q[0];
rz(1.5051069356113125) q[0];
cx q[0],q[15];
x q[0];
cx q[10],q[12];
cx q[12],q[4];
cx q[4],q[12];
cx q[12],q[4];
cx q[4],q[13];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[3],q[4];
rz(-pi/4) q[4];
cx q[1],q[4];
rz(pi/4) q[4];
cx q[3],q[4];
rz(-pi/4) q[4];
cx q[1],q[4];
cx q[1],q[3];
rz(pi) q[1];
rz(-pi/4) q[3];
cx q[1],q[3];
rz(pi/2) q[3];
rz(pi/4) q[4];
sx q[12];
rz(pi) q[13];
cx q[13],q[2];
sx q[2];
rz(2.877703412444273) q[2];
sx q[2];
rz(-pi) q[2];
cx q[13],q[2];
sx q[2];
sx q[13];
rz(pi/2) q[13];
cx q[9],q[13];
rz(-pi) q[13];
sx q[13];
rz(-pi) q[13];
rz(0.20072193648910355) q[15];
cx q[0],q[15];
rz(-3.046068472343113) q[0];
sx q[0];
cx q[0],q[3];
cx q[3],q[0];
rz(-pi) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[5],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
sx q[5];
rz(3*pi/4) q[5];
rz(1.8916662866629483) q[15];
sx q[15];
rz(-0.6446839764907857) q[15];
sx q[15];
rz(-1.8917581994216066) q[15];
cx q[15],q[4];
rz(-pi/4) q[4];
cx q[1],q[4];
rz(pi/4) q[4];
cx q[15],q[4];
rz(-pi/4) q[4];
cx q[1],q[4];
cx q[1],q[15];
rz(3*pi/4) q[4];
sx q[4];
rz(-1.2979899968857875) q[4];
sx q[4];
rz(3*pi/4) q[4];
rz(-pi/4) q[15];
cx q[1],q[15];
sx q[1];
rz(pi/2) q[1];
sx q[16];
rz(-0.49634098308782804) q[16];
sx q[16];
cx q[10],q[16];
x q[10];
rz(pi/2) q[10];
cx q[10],q[12];
sx q[10];
rz(-pi/2) q[10];
cx q[10],q[14];
x q[12];
rz(-pi) q[12];
cx q[12],q[2];
sx q[2];
cx q[9],q[2];
sx q[9];
rz(-pi/2) q[9];
sx q[12];
rz(-pi/4) q[12];
cx q[6],q[12];
rz(-pi/4) q[12];
cx q[14],q[10];
sx q[10];
rz(-2.597845526272513) q[10];
sx q[10];
rz(-1.1654641881756618) q[10];
rz(pi/2) q[14];
sx q[14];
cx q[11],q[14];
x q[11];
rz(3.751071038755452) q[11];
rz(-pi) q[14];
cx q[0],q[14];
cx q[14],q[0];
rz(pi/2) q[14];
sx q[14];
rz(-pi/2) q[14];
cx q[14],q[5];
rz(pi/4) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[0],q[5];
rz(pi/4) q[5];
cx q[3],q[5];
rz(-pi/4) q[5];
cx q[0],q[5];
rz(pi/4) q[5];
cx q[3],q[5];
rz(pi/4) q[5];
sx q[5];
rz(3*pi/4) q[5];
cx q[14],q[5];
rz(pi/4) q[5];
sx q[5];
rz(3*pi/2) q[5];
cx q[5],q[4];
rz(-2.84627949918999) q[4];
sx q[4];
rz(-3.04832521989575) q[4];
sx q[4];
rz(2.1066928309490764) q[4];
x q[5];
sx q[14];
rz(-pi) q[14];
cx q[15],q[12];
rz(pi/4) q[12];
cx q[6],q[12];
sx q[6];
rz(3*pi/4) q[6];
cx q[8],q[6];
rz(pi/4) q[6];
sx q[6];
cx q[11],q[8];
rz(-2.180274711960556) q[8];
cx q[11],q[8];
rz(1.0104377346039026) q[8];
sx q[11];
rz(pi/2) q[11];
rz(-3*pi/4) q[12];
sx q[12];
rz(-0.8438154266272662) q[12];
sx q[15];
cx q[10],q[15];
x q[10];
rz(-0.9142259747164401) q[15];
sx q[15];
rz(4.44348629651231) q[15];
sx q[15];
rz(5*pi/2) q[15];
x q[16];
cx q[16],q[7];
rz(2.460881369460738) q[7];
cx q[16],q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi) q[7];
cx q[7],q[1];
rz(-pi/4) q[1];
cx q[7],q[1];
rz(3*pi/4) q[1];
sx q[1];
rz(-5.2038718330651745) q[1];
cx q[1],q[2];
rz(-2.6501098009093087) q[2];
cx q[1],q[2];
sx q[1];
rz(3*pi/4) q[1];
rz(-2.062279179475381) q[2];
sx q[2];
rz(3*pi/4) q[2];
cx q[6],q[2];
rz(-pi/4) q[2];
cx q[3],q[2];
rz(pi/4) q[2];
cx q[6],q[2];
rz(-pi/4) q[2];
sx q[2];
rz(pi/4) q[2];
rz(3*pi/4) q[6];
cx q[6],q[4];
rz(-pi/4) q[4];
cx q[12],q[3];
cx q[3],q[12];
rz(0.8325868555718422) q[3];
cx q[3],q[5];
rz(-0.8325868555718422) q[5];
cx q[3],q[5];
rz(-2.3090057980179512) q[5];
cx q[14],q[2];
rz(pi/4) q[2];
sx q[2];
rz(0.07521948191998185) q[2];
sx q[14];
cx q[15],q[3];
rz(-0.4461484590699758) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi) q[16];
sx q[16];
rz(-pi/2) q[16];
cx q[16],q[13];
rz(6.181543494428037) q[13];
cx q[16],q[13];
rz(pi/2) q[13];
sx q[13];
rz(-pi) q[13];
cx q[7],q[13];
rz(-pi/4) q[13];
cx q[7],q[13];
cx q[0],q[7];
rz(-pi/4) q[7];
cx q[10],q[7];
rz(pi/4) q[7];
cx q[0],q[7];
rz(pi/4) q[0];
rz(-pi/4) q[7];
cx q[10],q[7];
rz(3*pi/4) q[7];
cx q[7],q[12];
cx q[10],q[0];
rz(-pi/4) q[0];
rz(5*pi/4) q[10];
cx q[10],q[0];
x q[10];
cx q[2],q[10];
rz(-0.36289858611676196) q[10];
cx q[2],q[10];
sx q[2];
rz(4.684428261021783) q[2];
sx q[2];
rz(7.6501236789685745) q[2];
rz(0.36289858611676173) q[10];
cx q[12],q[7];
cx q[7],q[12];
rz(pi/2) q[7];
rz(2.4763171629020073) q[12];
sx q[12];
rz(-1.613021354720944) q[12];
sx q[12];
rz(-pi/4) q[12];
rz(3*pi/4) q[13];
sx q[13];
rz(pi) q[13];
sx q[16];
rz(-pi) q[16];
cx q[16],q[9];
rz(5.222940161654445) q[9];
cx q[16],q[9];
cx q[13],q[9];
rz(4.32313949166098) q[9];
cx q[13],q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
sx q[13];
rz(pi/2) q[13];
cx q[16],q[1];
rz(3*pi/4) q[1];
sx q[1];
rz(-3*pi/4) q[1];
cx q[13],q[1];
rz(-pi/4) q[1];
cx q[0],q[1];
cx q[0],q[4];
rz(pi/4) q[1];
rz(pi/4) q[4];
cx q[6],q[4];
rz(-pi/4) q[4];
cx q[0],q[4];
cx q[0],q[6];
rz(3*pi/4) q[0];
rz(3*pi/4) q[4];
sx q[4];
rz(3*pi/4) q[4];
rz(-pi/4) q[6];
cx q[0],q[6];
rz(3*pi/4) q[6];
sx q[6];
rz(-pi/2) q[6];
cx q[6],q[3];
rz(1.9164065388485196) q[3];
cx q[6],q[3];
sx q[3];
rz(-0.5777040261303306) q[3];
sx q[3];
rz(pi/2) q[3];
sx q[6];
cx q[13],q[1];
rz(pi/4) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[9],q[1];
cx q[1],q[9];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[4],q[1];
rz(-pi/4) q[1];
rz(-3*pi/2) q[9];
sx q[9];
rz(3*pi/4) q[9];
cx q[8],q[9];
rz(-pi/4) q[9];
sx q[13];
rz(pi/2) q[13];
cx q[14],q[1];
rz(pi/4) q[1];
cx q[4],q[1];
rz(-pi/4) q[1];
cx q[14],q[1];
rz(3*pi/4) q[1];
sx q[1];
rz(-11*pi/16) q[1];
cx q[14],q[4];
rz(-pi/4) q[4];
rz(3*pi/4) q[14];
cx q[14],q[4];
sx q[14];
rz(-1.2345641865721522) q[14];
cx q[16],q[11];
rz(5.9191545824792) q[11];
cx q[16],q[11];
rz(-pi) q[11];
sx q[11];
rz(-pi/2) q[11];
cx q[11],q[13];
rz(4.855308754757299) q[13];
cx q[11],q[13];
sx q[11];
rz(-0.39581870818525444) q[11];
cx q[11],q[9];
rz(pi/4) q[9];
cx q[8],q[9];
cx q[8],q[12];
cx q[8],q[2];
rz(pi/4) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[1],q[9];
rz(-pi) q[9];
sx q[9];
rz(0.4992040690245969) q[9];
sx q[9];
cx q[11],q[4];
rz(-0.8898049043216808) q[4];
cx q[11],q[4];
rz(0.8898049043216808) q[4];
x q[4];
rz(-3.062557609332143) q[12];
sx q[12];
rz(-0.7404851041505491) q[12];
sx q[12];
rz(0.40119508057397724) q[12];
cx q[4],q[12];
rz(-pi) q[12];
rz(-pi/2) q[13];
cx q[7],q[13];
x q[7];
cx q[7],q[15];
cx q[14],q[9];
sx q[9];
rz(0.49920406902459646) q[9];
sx q[9];
rz(-pi) q[9];
cx q[14],q[9];
rz(pi/2) q[9];
sx q[9];
rz(-3.060543065582239) q[9];
sx q[14];
rz(-pi/2) q[14];
rz(4.172100840595829) q[15];
cx q[7],q[15];
cx q[15],q[2];
rz(pi/4) q[2];
cx q[7],q[2];
rz(-pi/4) q[2];
cx q[15],q[2];
rz(pi/4) q[2];
cx q[7],q[2];
rz(pi/4) q[2];
sx q[2];
rz(3*pi/4) q[2];
cx q[7],q[9];
cx q[8],q[2];
rz(pi/4) q[2];
sx q[2];
rz(4.630229387305588) q[2];
x q[8];
rz(-1.5420850964849198) q[8];
cx q[6],q[8];
rz(-pi) q[6];
sx q[6];
rz(3.027501495607037) q[6];
sx q[6];
sx q[8];
rz(3.027501495607037) q[8];
sx q[8];
rz(-pi) q[8];
cx q[6],q[8];
rz(pi/2) q[6];
sx q[6];
rz(-1.5995075571048734) q[8];
rz(-1.6518459148024505) q[9];
cx q[7],q[9];
rz(pi/2) q[9];
sx q[9];
rz(3*pi/4) q[9];
cx q[9],q[4];
rz(-pi/4) q[4];
cx q[7],q[4];
rz(pi/4) q[4];
cx q[9],q[4];
rz(-pi/4) q[4];
cx q[7],q[4];
rz(-0.549472196068943) q[4];
cx q[7],q[9];
rz(-pi/4) q[7];
rz(-pi/4) q[9];
cx q[7],q[9];
cx q[14],q[2];
sx q[2];
rz(2.8771305687065745) q[2];
sx q[2];
rz(-pi) q[2];
sx q[14];
rz(2.8771305687065745) q[14];
sx q[14];
cx q[14],q[2];
x q[2];
rz(3.428461354099877) q[2];
rz(-3*pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
x q[16];
cx q[16],q[10];
rz(-pi/4) q[10];
cx q[0],q[10];
rz(pi/4) q[10];
cx q[16],q[10];
rz(-pi/4) q[10];
cx q[0],q[10];
rz(1.8560015517948427) q[10];
sx q[10];
rz(-1.7446398345485017) q[10];
sx q[10];
rz(1.1270445819549089) q[10];
rz(pi/4) q[16];
cx q[0],q[16];
rz(-pi/4) q[16];
cx q[0],q[16];
cx q[0],q[5];
rz(-pi/4) q[5];
cx q[13],q[5];
rz(pi/4) q[5];
cx q[0],q[5];
rz(-pi/4) q[5];
cx q[13],q[5];
rz(3*pi/4) q[5];
sx q[5];
rz(-0.16251171825947464) q[5];
cx q[13],q[0];
rz(-pi/4) q[0];
rz(3*pi/4) q[13];
cx q[13],q[0];
rz(4.067218134896907) q[0];
sx q[0];
rz(3.3058579655595257) q[0];
sx q[0];
rz(15.205862675007264) q[0];
cx q[13],q[11];
rz(-1.8559690410828578) q[11];
cx q[13],q[11];
rz(1.3567629660669986) q[11];
sx q[11];
rz(-2.711080745938503) q[11];
sx q[11];
rz(1.8265147023856407) q[11];
sx q[13];
rz(pi/2) q[13];
cx q[1],q[13];
rz(-pi/16) q[13];
cx q[1],q[13];
cx q[1],q[15];
rz(pi/16) q[13];
rz(-pi/16) q[15];
cx q[15],q[13];
rz(pi/16) q[13];
cx q[15],q[13];
cx q[1],q[15];
rz(-pi/16) q[13];
rz(13*pi/16) q[15];
cx q[15],q[13];
rz(-pi/16) q[13];
cx q[15],q[13];
rz(pi/16) q[13];
cx q[15],q[0];
rz(-pi/16) q[0];
cx q[0],q[13];
rz(pi/16) q[13];
cx q[0],q[13];
cx q[1],q[0];
rz(pi/16) q[0];
rz(-pi/16) q[13];
cx q[0],q[13];
rz(-pi/16) q[13];
cx q[0],q[13];
rz(pi/16) q[13];
cx q[15],q[0];
rz(-pi/16) q[0];
cx q[0],q[13];
rz(pi/16) q[13];
cx q[0],q[13];
cx q[1],q[0];
rz(pi/16) q[0];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[12];
cx q[12],q[1];
rz(pi/2) q[12];
sx q[12];
rz(5.4678562357181315) q[12];
sx q[12];
rz(11*pi/4) q[12];
rz(-pi/16) q[13];
cx q[0],q[13];
rz(-pi/16) q[13];
cx q[0],q[13];
cx q[5],q[0];
rz(0.5139819104794644) q[0];
cx q[5],q[0];
rz(pi/2) q[0];
sx q[0];
cx q[4],q[5];
rz(-1.938188125805667) q[5];
sx q[5];
rz(-1.6091256477907159) q[5];
sx q[5];
cx q[4],q[5];
cx q[4],q[6];
sx q[5];
rz(-1.6091256477907159) q[5];
sx q[5];
rz(-0.832323998069386) q[5];
cx q[6],q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi) q[6];
sx q[6];
rz(-pi) q[6];
cx q[7],q[5];
rz(-pi) q[5];
sx q[5];
rz(-2.08839015956636) q[5];
cx q[2],q[5];
rz(-2.62399882081833) q[5];
cx q[2],q[5];
rz(-pi/2) q[5];
sx q[5];
rz(-0.6399782721272791) q[5];
sx q[7];
rz(pi/2) q[7];
cx q[11],q[6];
rz(-0.7059211512710498) q[6];
cx q[11],q[6];
rz(-0.07947701212639835) q[6];
sx q[11];
rz(-1.81853136875523) q[11];
sx q[11];
rz(9*pi/16) q[13];
sx q[13];
rz(3*pi/4) q[13];
cx q[13],q[3];
rz(-pi/4) q[3];
cx q[15],q[3];
rz(pi/4) q[3];
cx q[13],q[3];
rz(-pi/4) q[3];
cx q[15],q[3];
rz(3*pi/4) q[3];
sx q[3];
cx q[0],q[3];
rz(0.8230476330089902) q[3];
cx q[0],q[3];
sx q[0];
rz(-1.408130768410361) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[15],q[13];
rz(-pi/4) q[13];
cx q[15],q[13];
rz(2.926939029897203) q[13];
cx q[13],q[8];
rz(-pi/4) q[8];
cx q[13],q[8];
rz(pi/4) q[8];
cx q[14],q[8];
x q[14];
cx q[4],q[14];
sx q[4];
rz(-pi/4) q[4];
sx q[4];
rz(pi/2) q[14];
cx q[4],q[14];
sx q[4];
rz(-3.8795318890289203) q[4];
rz(-pi/2) q[14];
sx q[14];
rz(3*pi/4) q[14];
cx q[2],q[14];
sx q[2];
rz(pi) q[2];
rz(pi/4) q[14];
cx q[4],q[14];
rz(pi/4) q[14];
sx q[14];
rz(pi/2) q[14];
sx q[15];
rz(pi/2) q[15];
rz(-2.260653367280072) q[16];
sx q[16];
rz(4.740113349420723) q[16];
sx q[16];
rz(4.730018282590407) q[16];
cx q[16],q[10];
sx q[10];
rz(0.8327160273136465) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[9],q[10];
sx q[9];
rz(pi/4) q[9];
cx q[1],q[9];
rz(pi/4) q[9];
rz(pi/2) q[10];
sx q[10];
rz(-2.4035123541085426) q[10];
sx q[10];
cx q[10],q[9];
rz(-pi/4) q[9];
cx q[1],q[9];
rz(3*pi/4) q[9];
sx q[9];
rz(4.1829784569386845) q[9];
rz(pi/4) q[10];
cx q[1],q[10];
rz(0.7358776408470002) q[1];
rz(-pi/4) q[10];
cx q[1],q[10];
sx q[1];
rz(-0.5839502596637871) q[1];
sx q[1];
rz(-0.4885298982485846) q[1];
cx q[9],q[10];
sx q[9];
cx q[9],q[1];
rz(1.3185680826701562) q[1];
cx q[9],q[1];
rz(pi/2) q[1];
sx q[1];
rz(-pi) q[1];
cx q[6],q[1];
rz(-pi/4) q[1];
cx q[6],q[1];
rz(-3*pi/4) q[1];
sx q[1];
rz(3.127038183961658) q[1];
sx q[1];
sx q[6];
rz(pi/2) q[6];
sx q[9];
rz(-0.4803437789823235) q[9];
sx q[9];
rz(-pi/4) q[9];
cx q[16],q[15];
rz(-pi/4) q[15];
cx q[16],q[15];
rz(3*pi/4) q[15];
sx q[15];
rz(3*pi/4) q[15];
cx q[15],q[12];
rz(pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[16],q[12];
rz(pi/4) q[12];
cx q[10],q[12];
rz(-pi/4) q[12];
cx q[16],q[12];
rz(pi/4) q[12];
cx q[10],q[12];
rz(-0.7507495184453195) q[10];
cx q[7],q[10];
sx q[7];
rz(0.44593825394272724) q[7];
sx q[7];
rz(-pi/2) q[7];
sx q[10];
rz(0.44593825394272724) q[10];
sx q[10];
rz(-pi) q[10];
cx q[7],q[10];
sx q[7];
rz(3*pi/4) q[7];
rz(0.7507495184453186) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/4) q[12];
sx q[12];
rz(3*pi/4) q[12];
cx q[15],q[12];
rz(pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[3];
x q[3];
sx q[12];
rz(-3*pi/2) q[12];
cx q[15],q[13];
rz(-pi/4) q[13];
cx q[8],q[13];
rz(pi/4) q[13];
cx q[15],q[13];
rz(-pi/4) q[13];
cx q[8],q[13];
cx q[8],q[15];
rz(pi/4) q[8];
rz(3*pi/4) q[13];
cx q[0],q[13];
cx q[13],q[0];
rz(-3*pi/2) q[0];
sx q[0];
rz(3*pi/4) q[0];
cx q[4],q[0];
rz(-pi/4) q[0];
rz(-3*pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[7],q[13];
rz(-pi/4) q[13];
cx q[7],q[13];
x q[7];
rz(pi/4) q[13];
cx q[13],q[1];
sx q[1];
rz(3.127038183961658) q[1];
sx q[1];
rz(-pi) q[1];
cx q[13],q[1];
cx q[14],q[0];
rz(pi/4) q[0];
cx q[4],q[0];
rz(-pi/4) q[0];
sx q[4];
rz(8.494719734368612) q[4];
sx q[4];
rz(11.733513523003403) q[4];
rz(-pi/4) q[15];
cx q[8],q[15];
sx q[8];
rz(pi) q[8];
cx q[8],q[10];
rz(3.7309739743092494) q[10];
cx q[8],q[10];
sx q[8];
rz(-pi/2) q[8];
cx q[8],q[0];
rz(5.641023283172647) q[0];
cx q[8],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[8];
rz(pi/2) q[8];
sx q[10];
cx q[2],q[10];
rz(3.7805157781642875) q[10];
cx q[2],q[10];
sx q[2];
rz(-0.01451266804337692) q[2];
sx q[2];
rz(2.8127850672072325) q[2];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[14],q[10];
rz(1.2003775639103518) q[15];
sx q[15];
rz(5.779635601470118) q[15];
sx q[15];
rz(11.55470040454415) q[15];
sx q[16];
rz(0.8651547297681352) q[16];
cx q[12],q[16];
sx q[12];
rz(2.38413511831303) q[12];
sx q[12];
sx q[16];
rz(2.38413511831303) q[16];
sx q[16];
rz(-pi) q[16];
cx q[12],q[16];
rz(-3*pi/2) q[12];
sx q[12];
rz(-pi/2) q[12];
cx q[3],q[12];
rz(0.12705406750904324) q[12];
cx q[3],q[12];
cx q[12],q[9];
rz(pi/4) q[9];
sx q[9];
rz(-1.4707999582740392) q[16];
cx q[16],q[5];
rz(-0.6911563680662884) q[5];
sx q[5];
rz(-0.37335301441902224) q[5];
sx q[5];
cx q[16],q[5];
cx q[3],q[16];
sx q[5];
rz(-0.37335301441902224) q[5];
sx q[5];
rz(1.4425794993012397) q[5];
cx q[6],q[5];
sx q[5];
rz(0.6237217999833495) q[5];
sx q[5];
rz(-pi) q[5];
sx q[6];
rz(0.6237217999833495) q[6];
sx q[6];
rz(-5*pi/2) q[6];
cx q[6],q[5];
rz(-1.6822411859025697) q[5];
sx q[6];
rz(-pi/2) q[6];
rz(-1.2154495654283073) q[16];
cx q[3],q[16];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
measure q[10] -> meas[10];
measure q[11] -> meas[11];
measure q[12] -> meas[12];
measure q[13] -> meas[13];
measure q[14] -> meas[14];
measure q[15] -> meas[15];
measure q[16] -> meas[16];