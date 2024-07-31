OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg meas[20];
rz(-pi/2) q[0];
sx q[0];
rz(0.312036201003675) q[0];
rz(pi/2) q[1];
sx q[1];
rz(1.5708083110198021) q[1];
rz(pi/2) q[2];
sx q[2];
rz(1.5708202952447075) q[2];
rz(pi/2) q[3];
sx q[3];
rz(1.5708442636945181) q[3];
rz(pi/2) q[4];
sx q[4];
rz(1.5708922005941395) q[4];
rz(pi/2) q[5];
sx q[5];
rz(1.5709880743933822) q[5];
rz(pi/2) q[6];
sx q[6];
rz(1.571179821991868) q[6];
rz(pi/2) q[7];
sx q[7];
rz(1.5715633171888392) q[7];
rz(pi/2) q[8];
sx q[8];
rz(1.5723303075827821) q[8];
rz(pi/2) q[9];
sx q[9];
rz(1.573864288370668) q[9];
rz(pi/2) q[10];
sx q[10];
rz(1.576932249946439) q[10];
rz(pi/2) q[11];
sx q[11];
rz(1.5830681730979819) q[11];
rz(pi/2) q[12];
sx q[12];
rz(1.595340019401067) q[12];
rz(pi/2) q[13];
sx q[13];
rz(1.6198837120072371) q[13];
rz(pi/2) q[14];
sx q[14];
rz(1.6689710972195775) q[14];
rz(pi/2) q[15];
sx q[15];
rz(9*pi/16) q[15];
rz(pi/2) q[16];
sx q[16];
rz(5*pi/8) q[16];
rz(pi/2) q[17];
sx q[17];
rz(3*pi/4) q[17];
rz(pi/2) q[18];
sx q[18];
rz(pi) q[18];
sx q[19];
rz(pi/2) q[19];
cx q[0],q[19];
x q[0];
rz(0.9272952180016127) q[19];
cx q[0],q[19];
rz(-1.2587541336787682) q[0];
rz(-0.28379410920832715) q[19];
sx q[19];
cx q[1],q[19];
sx q[19];
rz(1.2870022175865685) q[19];
sx q[19];
rz(-pi) q[19];
cx q[1],q[19];
rz(-pi) q[19];
sx q[19];
rz(1.2870022175865685) q[19];
sx q[19];
cx q[2],q[19];
rz(-pi) q[19];
sx q[19];
rz(0.5675882184166556) q[19];
sx q[19];
cx q[2],q[19];
sx q[19];
rz(0.5675882184166556) q[19];
sx q[19];
rz(-pi) q[19];
cx q[3],q[19];
sx q[19];
rz(2.006416216756482) q[19];
sx q[19];
rz(-pi) q[19];
cx q[3],q[19];
rz(-pi) q[19];
sx q[19];
rz(2.006416216756481) q[19];
sx q[19];
cx q[4],q[19];
sx q[19];
rz(0.8712397799231706) q[19];
sx q[19];
rz(-pi) q[19];
cx q[4],q[19];
rz(-pi) q[19];
sx q[19];
rz(0.8712397799231724) q[19];
sx q[19];
cx q[5],q[19];
rz(-pi) q[19];
sx q[19];
rz(1.399113093743451) q[19];
sx q[19];
cx q[5],q[19];
sx q[19];
rz(1.399113093743451) q[19];
sx q[19];
rz(-pi) q[19];
cx q[6],q[19];
sx q[19];
rz(0.34336646610288746) q[19];
sx q[19];
rz(-pi) q[19];
cx q[6],q[19];
rz(-pi) q[19];
sx q[19];
rz(0.343366466102895) q[19];
sx q[19];
cx q[7],q[19];
rz(-pi) q[19];
sx q[19];
rz(2.4548597213840058) q[19];
sx q[19];
cx q[7],q[19];
sx q[19];
rz(2.454859721384013) q[19];
sx q[19];
rz(-pi) q[19];
cx q[8],q[19];
rz(-pi) q[19];
sx q[19];
rz(1.768126789178238) q[19];
sx q[19];
cx q[8],q[19];
sx q[19];
rz(1.7681267891782166) q[19];
sx q[19];
rz(-pi) q[19];
cx q[9],q[19];
rz(-pi) q[19];
sx q[19];
rz(0.39466092476667125) q[19];
sx q[19];
cx q[9],q[19];
sx q[19];
rz(0.39466092476664993) q[19];
sx q[19];
rz(-pi) q[19];
cx q[10],q[19];
sx q[19];
rz(2.3522708040564613) q[19];
sx q[19];
rz(-pi) q[19];
cx q[10],q[19];
rz(-pi) q[19];
sx q[19];
rz(2.3522708040564826) q[19];
sx q[19];
cx q[11],q[19];
sx q[19];
rz(1.5629489545232529) q[19];
sx q[19];
rz(-pi) q[19];
cx q[11],q[19];
rz(-pi) q[19];
sx q[19];
rz(1.5629489545230477) q[19];
sx q[19];
cx q[12],q[19];
rz(-pi) q[19];
sx q[19];
rz(0.0156947445436173) q[19];
sx q[19];
cx q[12],q[19];
sx q[19];
rz(0.015694744543368166) q[19];
sx q[19];
rz(-pi) q[19];
cx q[13],q[19];
sx q[19];
rz(3.1102031645031385) q[19];
sx q[19];
rz(-pi) q[19];
cx q[13],q[19];
rz(-pi) q[19];
sx q[19];
rz(3.1102031645024777) q[19];
sx q[19];
cx q[14],q[19];
sx q[19];
rz(3.0788136754161517) q[19];
sx q[19];
rz(-pi) q[19];
cx q[14],q[19];
rz(-pi) q[19];
sx q[19];
rz(3.078813675415492) q[19];
sx q[19];
cx q[15],q[19];
sx q[19];
rz(3.0160346972403627) q[19];
sx q[19];
rz(-pi) q[19];
cx q[15],q[19];
rz(-pi) q[19];
sx q[19];
rz(3.01603469724334) q[19];
sx q[19];
cx q[16],q[19];
sx q[19];
rz(2.890476740896058) q[19];
sx q[19];
rz(-pi) q[19];
cx q[16],q[19];
rz(-pi) q[19];
sx q[19];
rz(2.89047674089176) q[19];
sx q[19];
cx q[17],q[19];
sx q[19];
rz(2.6393608281928973) q[19];
sx q[19];
rz(-pi) q[19];
cx q[17],q[19];
rz(-pi) q[19];
sx q[19];
rz(2.6393608282031513) q[19];
sx q[19];
cx q[18],q[19];
sx q[19];
rz(2.137129002801129) q[19];
sx q[19];
rz(-pi) q[19];
cx q[18],q[19];
sx q[18];
rz(pi/2) q[18];
cx q[17],q[18];
rz(pi/4) q[18];
cx q[17],q[18];
sx q[17];
rz(pi/2) q[17];
rz(-pi/4) q[18];
cx q[16],q[18];
rz(pi/8) q[18];
cx q[16],q[18];
cx q[16],q[17];
rz(pi/4) q[17];
cx q[16],q[17];
sx q[16];
rz(pi/2) q[16];
rz(-pi/4) q[17];
rz(-pi/8) q[18];
cx q[15],q[18];
rz(pi/16) q[18];
cx q[15],q[18];
cx q[15],q[17];
rz(pi/8) q[17];
cx q[15],q[17];
cx q[15],q[16];
rz(pi/4) q[16];
cx q[15],q[16];
sx q[15];
rz(pi/2) q[15];
rz(-pi/4) q[16];
rz(-pi/8) q[17];
rz(-pi/16) q[18];
cx q[14],q[18];
rz(pi/32) q[18];
cx q[14],q[18];
cx q[14],q[17];
rz(pi/16) q[17];
cx q[14],q[17];
cx q[14],q[16];
rz(pi/8) q[16];
cx q[14],q[16];
cx q[14],q[15];
rz(pi/4) q[15];
cx q[14],q[15];
sx q[14];
rz(pi/2) q[14];
rz(-pi/4) q[15];
rz(-pi/8) q[16];
rz(-pi/16) q[17];
rz(-pi/32) q[18];
cx q[13],q[18];
rz(pi/64) q[18];
cx q[13],q[18];
cx q[13],q[17];
rz(pi/32) q[17];
cx q[13],q[17];
cx q[13],q[16];
rz(pi/16) q[16];
cx q[13],q[16];
cx q[13],q[15];
rz(pi/8) q[15];
cx q[13],q[15];
cx q[13],q[14];
rz(pi/4) q[14];
cx q[13],q[14];
sx q[13];
rz(pi/2) q[13];
rz(-pi/4) q[14];
rz(-pi/8) q[15];
rz(-pi/16) q[16];
rz(-pi/32) q[17];
rz(-pi/64) q[18];
cx q[12],q[18];
rz(pi/128) q[18];
cx q[12],q[18];
cx q[12],q[17];
rz(pi/64) q[17];
cx q[12],q[17];
cx q[12],q[16];
rz(pi/32) q[16];
cx q[12],q[16];
cx q[12],q[15];
rz(pi/16) q[15];
cx q[12],q[15];
cx q[12],q[14];
rz(pi/8) q[14];
cx q[12],q[14];
cx q[12],q[13];
rz(pi/4) q[13];
cx q[12],q[13];
sx q[12];
rz(pi/2) q[12];
rz(-pi/4) q[13];
rz(-pi/8) q[14];
rz(-pi/16) q[15];
rz(-pi/32) q[16];
rz(-pi/64) q[17];
rz(-pi/128) q[18];
cx q[11],q[18];
rz(pi/256) q[18];
cx q[11],q[18];
cx q[11],q[17];
rz(pi/128) q[17];
cx q[11],q[17];
cx q[11],q[16];
rz(pi/64) q[16];
cx q[11],q[16];
cx q[11],q[15];
rz(pi/32) q[15];
cx q[11],q[15];
cx q[11],q[14];
rz(pi/16) q[14];
cx q[11],q[14];
cx q[11],q[13];
rz(pi/8) q[13];
cx q[11],q[13];
cx q[11],q[12];
rz(pi/4) q[12];
cx q[11],q[12];
sx q[11];
rz(pi/2) q[11];
rz(-pi/4) q[12];
rz(-pi/8) q[13];
rz(-pi/16) q[14];
rz(-pi/32) q[15];
rz(-pi/64) q[16];
rz(-pi/128) q[17];
rz(-pi/256) q[18];
cx q[10],q[18];
rz(pi/512) q[18];
cx q[10],q[18];
cx q[10],q[17];
rz(pi/256) q[17];
cx q[10],q[17];
cx q[10],q[16];
rz(pi/128) q[16];
cx q[10],q[16];
cx q[10],q[15];
rz(pi/64) q[15];
cx q[10],q[15];
cx q[10],q[14];
rz(pi/32) q[14];
cx q[10],q[14];
cx q[10],q[13];
rz(pi/16) q[13];
cx q[10],q[13];
cx q[10],q[12];
rz(pi/8) q[12];
cx q[10],q[12];
cx q[10],q[11];
rz(pi/4) q[11];
cx q[10],q[11];
sx q[10];
rz(pi/2) q[10];
rz(-pi/4) q[11];
rz(-pi/8) q[12];
rz(-pi/16) q[13];
rz(-pi/32) q[14];
rz(-pi/64) q[15];
rz(-pi/128) q[16];
rz(-pi/256) q[17];
rz(-pi/512) q[18];
cx q[9],q[18];
rz(pi/1024) q[18];
cx q[9],q[18];
cx q[9],q[17];
rz(pi/512) q[17];
cx q[9],q[17];
cx q[9],q[16];
rz(pi/256) q[16];
cx q[9],q[16];
cx q[9],q[15];
rz(pi/128) q[15];
cx q[9],q[15];
cx q[9],q[14];
rz(pi/64) q[14];
cx q[9],q[14];
cx q[9],q[13];
rz(pi/32) q[13];
cx q[9],q[13];
cx q[9],q[12];
rz(pi/16) q[12];
cx q[9],q[12];
cx q[9],q[11];
rz(pi/8) q[11];
cx q[9],q[11];
cx q[9],q[10];
rz(pi/4) q[10];
cx q[9],q[10];
sx q[9];
rz(pi/2) q[9];
rz(-pi/4) q[10];
rz(-pi/8) q[11];
rz(-pi/16) q[12];
rz(-pi/32) q[13];
rz(-pi/64) q[14];
rz(-pi/128) q[15];
rz(-pi/256) q[16];
rz(-pi/512) q[17];
rz(-pi/1024) q[18];
cx q[8],q[18];
rz(pi/2048) q[18];
cx q[8],q[18];
cx q[8],q[17];
rz(pi/1024) q[17];
cx q[8],q[17];
cx q[8],q[16];
rz(pi/512) q[16];
cx q[8],q[16];
cx q[8],q[15];
rz(pi/256) q[15];
cx q[8],q[15];
cx q[8],q[14];
rz(pi/128) q[14];
cx q[8],q[14];
cx q[8],q[13];
rz(pi/64) q[13];
cx q[8],q[13];
cx q[8],q[12];
rz(pi/32) q[12];
cx q[8],q[12];
cx q[8],q[11];
rz(pi/16) q[11];
cx q[8],q[11];
cx q[8],q[10];
rz(pi/8) q[10];
cx q[8],q[10];
cx q[8],q[9];
rz(pi/4) q[9];
cx q[8],q[9];
sx q[8];
rz(pi/2) q[8];
rz(-pi/4) q[9];
rz(-pi/8) q[10];
rz(-pi/16) q[11];
rz(-pi/32) q[12];
rz(-pi/64) q[13];
rz(-pi/128) q[14];
rz(-pi/256) q[15];
rz(-pi/512) q[16];
rz(-pi/1024) q[17];
rz(-pi/2048) q[18];
cx q[7],q[18];
rz(pi/4096) q[18];
cx q[7],q[18];
cx q[7],q[17];
rz(pi/2048) q[17];
cx q[7],q[17];
cx q[7],q[16];
rz(pi/1024) q[16];
cx q[7],q[16];
cx q[7],q[15];
rz(pi/512) q[15];
cx q[7],q[15];
cx q[7],q[14];
rz(pi/256) q[14];
cx q[7],q[14];
cx q[7],q[13];
rz(pi/128) q[13];
cx q[7],q[13];
cx q[7],q[12];
rz(pi/64) q[12];
cx q[7],q[12];
cx q[7],q[11];
rz(pi/32) q[11];
cx q[7],q[11];
cx q[7],q[10];
rz(pi/16) q[10];
cx q[7],q[10];
cx q[7],q[9];
rz(pi/8) q[9];
cx q[7],q[9];
cx q[7],q[8];
rz(pi/4) q[8];
cx q[7],q[8];
sx q[7];
rz(pi/2) q[7];
rz(-pi/4) q[8];
rz(-pi/8) q[9];
rz(-pi/16) q[10];
rz(-pi/32) q[11];
rz(-pi/64) q[12];
rz(-pi/128) q[13];
rz(-pi/256) q[14];
rz(-pi/512) q[15];
rz(-pi/1024) q[16];
rz(-pi/2048) q[17];
rz(-pi/4096) q[18];
cx q[6],q[18];
rz(pi/8192) q[18];
cx q[6],q[18];
cx q[6],q[17];
rz(pi/4096) q[17];
cx q[6],q[17];
cx q[6],q[16];
rz(pi/2048) q[16];
cx q[6],q[16];
cx q[6],q[15];
rz(pi/1024) q[15];
cx q[6],q[15];
cx q[6],q[14];
rz(pi/512) q[14];
cx q[6],q[14];
cx q[6],q[13];
rz(pi/256) q[13];
cx q[6],q[13];
cx q[6],q[12];
rz(pi/128) q[12];
cx q[6],q[12];
cx q[6],q[11];
rz(pi/64) q[11];
cx q[6],q[11];
cx q[6],q[10];
rz(pi/32) q[10];
cx q[6],q[10];
cx q[6],q[9];
rz(pi/16) q[9];
cx q[6],q[9];
cx q[6],q[8];
rz(pi/8) q[8];
cx q[6],q[8];
cx q[6],q[7];
rz(pi/4) q[7];
cx q[6],q[7];
sx q[6];
rz(pi/2) q[6];
rz(-pi/4) q[7];
rz(-pi/8) q[8];
rz(-pi/16) q[9];
rz(-pi/32) q[10];
rz(-pi/64) q[11];
rz(-pi/128) q[12];
rz(-pi/256) q[13];
rz(-pi/512) q[14];
rz(-pi/1024) q[15];
rz(-pi/2048) q[16];
rz(-pi/4096) q[17];
rz(-pi/8192) q[18];
cx q[5],q[18];
rz(pi/16384) q[18];
cx q[5],q[18];
cx q[5],q[17];
rz(pi/8192) q[17];
cx q[5],q[17];
cx q[5],q[16];
rz(pi/4096) q[16];
cx q[5],q[16];
cx q[5],q[15];
rz(pi/2048) q[15];
cx q[5],q[15];
cx q[5],q[14];
rz(pi/1024) q[14];
cx q[5],q[14];
cx q[5],q[13];
rz(pi/512) q[13];
cx q[5],q[13];
cx q[5],q[12];
rz(pi/256) q[12];
cx q[5],q[12];
cx q[5],q[11];
rz(pi/128) q[11];
cx q[5],q[11];
cx q[5],q[10];
rz(pi/64) q[10];
cx q[5],q[10];
cx q[5],q[9];
rz(pi/32) q[9];
cx q[5],q[9];
cx q[5],q[8];
rz(pi/16) q[8];
cx q[5],q[8];
cx q[5],q[7];
rz(pi/8) q[7];
cx q[5],q[7];
cx q[5],q[6];
rz(pi/4) q[6];
cx q[5],q[6];
sx q[5];
rz(pi/2) q[5];
rz(-pi/4) q[6];
rz(-pi/8) q[7];
rz(-pi/16) q[8];
rz(-pi/32) q[9];
rz(-pi/64) q[10];
rz(-pi/128) q[11];
rz(-pi/256) q[12];
rz(-pi/512) q[13];
rz(-pi/1024) q[14];
rz(-pi/2048) q[15];
rz(-pi/4096) q[16];
rz(-pi/8192) q[17];
rz(-pi/16384) q[18];
cx q[4],q[18];
rz(pi/32768) q[18];
cx q[4],q[18];
cx q[4],q[17];
rz(pi/16384) q[17];
cx q[4],q[17];
cx q[4],q[16];
rz(pi/8192) q[16];
cx q[4],q[16];
cx q[4],q[15];
rz(pi/4096) q[15];
cx q[4],q[15];
cx q[4],q[14];
rz(pi/2048) q[14];
cx q[4],q[14];
cx q[4],q[13];
rz(pi/1024) q[13];
cx q[4],q[13];
cx q[4],q[12];
rz(pi/512) q[12];
cx q[4],q[12];
cx q[4],q[11];
rz(pi/256) q[11];
cx q[4],q[11];
cx q[4],q[10];
rz(pi/128) q[10];
cx q[4],q[10];
cx q[4],q[9];
rz(pi/64) q[9];
cx q[4],q[9];
cx q[4],q[8];
rz(pi/32) q[8];
cx q[4],q[8];
cx q[4],q[7];
rz(pi/16) q[7];
cx q[4],q[7];
cx q[4],q[6];
rz(pi/8) q[6];
cx q[4],q[6];
cx q[4],q[5];
rz(pi/4) q[5];
cx q[4],q[5];
sx q[4];
rz(pi/2) q[4];
rz(-pi/4) q[5];
rz(-pi/8) q[6];
rz(-pi/16) q[7];
rz(-pi/32) q[8];
rz(-pi/64) q[9];
rz(-pi/128) q[10];
rz(-pi/256) q[11];
rz(-pi/512) q[12];
rz(-pi/1024) q[13];
rz(-pi/2048) q[14];
rz(-pi/4096) q[15];
rz(-pi/8192) q[16];
rz(-pi/16384) q[17];
rz(-pi/32768) q[18];
cx q[3],q[18];
rz(pi/65536) q[18];
cx q[3],q[18];
cx q[3],q[17];
rz(pi/32768) q[17];
cx q[3],q[17];
cx q[3],q[16];
rz(pi/16384) q[16];
cx q[3],q[16];
cx q[3],q[15];
rz(pi/8192) q[15];
cx q[3],q[15];
cx q[3],q[14];
rz(pi/4096) q[14];
cx q[3],q[14];
cx q[3],q[13];
rz(pi/2048) q[13];
cx q[3],q[13];
cx q[3],q[12];
rz(pi/1024) q[12];
cx q[3],q[12];
cx q[3],q[11];
rz(pi/512) q[11];
cx q[3],q[11];
cx q[3],q[10];
rz(pi/256) q[10];
cx q[3],q[10];
cx q[3],q[9];
rz(pi/128) q[9];
cx q[3],q[9];
cx q[3],q[8];
rz(pi/64) q[8];
cx q[3],q[8];
cx q[3],q[7];
rz(pi/32) q[7];
cx q[3],q[7];
cx q[3],q[6];
rz(pi/16) q[6];
cx q[3],q[6];
cx q[3],q[5];
rz(pi/8) q[5];
cx q[3],q[5];
cx q[3],q[4];
rz(pi/4) q[4];
cx q[3],q[4];
sx q[3];
rz(pi/2) q[3];
rz(-pi/4) q[4];
rz(-pi/8) q[5];
rz(-pi/16) q[6];
rz(-pi/32) q[7];
rz(-pi/64) q[8];
rz(-pi/128) q[9];
rz(-pi/256) q[10];
rz(-pi/512) q[11];
rz(-pi/1024) q[12];
rz(-pi/2048) q[13];
rz(-pi/4096) q[14];
rz(-pi/8192) q[15];
rz(-pi/16384) q[16];
rz(-pi/32768) q[17];
rz(-pi/65536) q[18];
cx q[2],q[18];
rz(pi/131072) q[18];
cx q[2],q[18];
cx q[2],q[17];
rz(pi/65536) q[17];
cx q[2],q[17];
cx q[2],q[16];
rz(pi/32768) q[16];
cx q[2],q[16];
cx q[2],q[15];
rz(pi/16384) q[15];
cx q[2],q[15];
cx q[2],q[14];
rz(pi/8192) q[14];
cx q[2],q[14];
cx q[2],q[13];
rz(pi/4096) q[13];
cx q[2],q[13];
cx q[2],q[12];
rz(pi/2048) q[12];
cx q[2],q[12];
cx q[2],q[11];
rz(pi/1024) q[11];
cx q[2],q[11];
cx q[2],q[10];
rz(pi/512) q[10];
cx q[2],q[10];
cx q[2],q[9];
rz(pi/256) q[9];
cx q[2],q[9];
cx q[2],q[8];
rz(pi/128) q[8];
cx q[2],q[8];
cx q[2],q[7];
rz(pi/64) q[7];
cx q[2],q[7];
cx q[2],q[6];
rz(pi/32) q[6];
cx q[2],q[6];
cx q[2],q[5];
rz(pi/16) q[5];
cx q[2],q[5];
cx q[2],q[4];
rz(pi/8) q[4];
cx q[2],q[4];
cx q[2],q[3];
rz(pi/4) q[3];
cx q[2],q[3];
sx q[2];
rz(pi/2) q[2];
rz(-pi/4) q[3];
rz(-pi/8) q[4];
rz(-pi/16) q[5];
rz(-pi/32) q[6];
rz(-pi/64) q[7];
rz(-pi/128) q[8];
rz(-pi/256) q[9];
rz(-pi/512) q[10];
rz(-pi/1024) q[11];
rz(-pi/2048) q[12];
rz(-pi/4096) q[13];
rz(-pi/8192) q[14];
rz(-pi/16384) q[15];
rz(-pi/32768) q[16];
rz(-pi/65536) q[17];
rz(-pi/131072) q[18];
cx q[1],q[18];
rz(pi/262144) q[18];
cx q[1],q[18];
cx q[1],q[17];
rz(pi/131072) q[17];
cx q[1],q[17];
cx q[1],q[16];
rz(pi/65536) q[16];
cx q[1],q[16];
cx q[1],q[15];
rz(pi/32768) q[15];
cx q[1],q[15];
cx q[1],q[14];
rz(pi/16384) q[14];
cx q[1],q[14];
cx q[1],q[13];
rz(pi/8192) q[13];
cx q[1],q[13];
cx q[1],q[12];
rz(pi/4096) q[12];
cx q[1],q[12];
cx q[1],q[11];
rz(pi/2048) q[11];
cx q[1],q[11];
cx q[1],q[10];
rz(pi/1024) q[10];
cx q[1],q[10];
cx q[1],q[9];
rz(pi/512) q[9];
cx q[1],q[9];
cx q[1],q[8];
rz(pi/256) q[8];
cx q[1],q[8];
cx q[1],q[7];
rz(pi/128) q[7];
cx q[1],q[7];
cx q[1],q[6];
rz(pi/64) q[6];
cx q[1],q[6];
cx q[1],q[5];
rz(pi/32) q[5];
cx q[1],q[5];
cx q[1],q[4];
rz(pi/16) q[4];
cx q[1],q[4];
cx q[1],q[3];
rz(pi/8) q[3];
cx q[1],q[3];
cx q[1],q[2];
rz(pi/4) q[2];
cx q[1],q[2];
sx q[1];
rz(pi/2) q[1];
rz(-pi/4) q[2];
rz(-pi/8) q[3];
rz(-pi/16) q[4];
rz(-pi/32) q[5];
rz(-pi/64) q[6];
rz(-pi/128) q[7];
rz(-pi/256) q[8];
rz(-pi/512) q[9];
rz(-pi/1024) q[10];
rz(-pi/2048) q[11];
rz(-pi/4096) q[12];
rz(-pi/8192) q[13];
rz(-pi/16384) q[14];
rz(-pi/32768) q[15];
rz(-pi/65536) q[16];
rz(-pi/131072) q[17];
cx q[0],q[17];
rz(pi/262144) q[17];
cx q[0],q[17];
cx q[0],q[16];
rz(pi/131072) q[16];
cx q[0],q[16];
cx q[0],q[15];
rz(pi/65536) q[15];
cx q[0],q[15];
cx q[0],q[14];
rz(pi/32768) q[14];
cx q[0],q[14];
cx q[0],q[13];
rz(pi/16384) q[13];
cx q[0],q[13];
cx q[0],q[12];
rz(pi/8192) q[12];
cx q[0],q[12];
cx q[0],q[11];
rz(pi/4096) q[11];
cx q[0],q[11];
cx q[0],q[10];
rz(pi/2048) q[10];
cx q[0],q[10];
cx q[0],q[9];
rz(pi/1024) q[9];
cx q[0],q[9];
cx q[0],q[8];
rz(pi/512) q[8];
cx q[0],q[8];
cx q[0],q[7];
rz(pi/256) q[7];
cx q[0],q[7];
cx q[0],q[6];
rz(pi/128) q[6];
cx q[0],q[6];
cx q[0],q[5];
rz(pi/64) q[5];
cx q[0],q[5];
cx q[0],q[4];
rz(pi/32) q[4];
cx q[0],q[4];
cx q[0],q[3];
rz(pi/16) q[3];
cx q[0],q[3];
cx q[0],q[2];
rz(pi/8) q[2];
cx q[0],q[2];
cx q[0],q[1];
rz(pi/4) q[1];
cx q[0],q[1];
sx q[0];
rz(pi/2) q[0];
rz(-pi/4) q[1];
rz(-pi/8) q[2];
rz(-pi/16) q[3];
rz(-pi/32) q[4];
rz(-pi/64) q[5];
rz(-pi/128) q[6];
rz(-pi/256) q[7];
rz(-pi/512) q[8];
rz(-pi/1024) q[9];
rz(-pi/2048) q[10];
rz(-pi/4096) q[11];
rz(-pi/8192) q[12];
rz(-pi/16384) q[13];
rz(-pi/32768) q[14];
rz(-pi/65536) q[15];
rz(-pi/131072) q[16];
rz(-pi/262144) q[17];
rz(-pi/262144) q[18];
rz(-pi) q[19];
sx q[19];
rz(2.137129002811382) q[19];
sx q[19];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19];
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
measure q[17] -> meas[17];
measure q[18] -> meas[18];
measure q[19] -> meas[19];
