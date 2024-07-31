OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg meas[16];
u2(0,-pi) q[0];
u2(0,-pi) q[1];
u2(0,-pi) q[2];
u2(0,-pi) q[3];
u2(0,-pi) q[4];
u2(0,-pi) q[5];
u2(0,-pi) q[6];
u2(0,-pi) q[7];
u2(0,-pi) q[8];
u2(0,-pi) q[9];
u2(0,-pi) q[10];
u2(0,-pi) q[11];
u2(0,-pi) q[12];
u2(0,-pi) q[13];
u2(0,-pi) q[14];
u3(0.9272952180016122,0,0) q[15];
cx q[0],q[15];
u(-0.9272952180016122,0,0) q[15];
cx q[0],q[15];
u3(0.9272952180016122,0,0) q[15];
cx q[1],q[15];
u(-1.8545904360032244,0,0) q[15];
cx q[1],q[15];
u3(1.8545904360032244,0,0) q[15];
cx q[2],q[15];
u(-3.7091808720064487,0,0) q[15];
cx q[2],q[15];
u3(2.574004435173138,-pi,-pi) q[15];
cx q[3],q[15];
u(-7.4183617440128975,0,0) q[15];
cx q[3],q[15];
u3(1.135176436833311,0,0) q[15];
cx q[4],q[15];
u(-14.836723488025795,0,0) q[15];
cx q[4],q[15];
u3(2.270352873666622,0,0) q[15];
cx q[5],q[15];
u(-29.67344697605159,0,0) q[15];
cx q[5],q[15];
u3(1.7424795598463425,-pi,-pi) q[15];
cx q[6],q[15];
u(-59.34689395210318,0,0) q[15];
cx q[6],q[15];
u3(2.7982261874869017,0,0) q[15];
cx q[7],q[15];
u(-118.69378790420636,0,0) q[15];
cx q[7],q[15];
u3(0.6867329322057831,-pi,-pi) q[15];
cx q[8],q[15];
u(-237.38757580841272,0,0) q[15];
cx q[8],q[15];
u3(1.3734658644115663,-pi,-pi) q[15];
cx q[9],q[15];
u(-474.77515161682544,0,0) q[15];
cx q[9],q[15];
u3(2.7469317288231325,-pi,-pi) q[15];
cx q[10],q[15];
u(-949.5503032336509,0,0) q[15];
cx q[10],q[15];
u3(0.7893218495333215,0,0) q[15];
cx q[11],q[15];
u(-1899.1006064673018,0,0) q[15];
cx q[11],q[15];
u3(1.5786436990666428,0,0) q[15];
cx q[12],q[15];
u(-3798.2012129346035,0,0) q[15];
cx q[12],q[15];
u3(3.1258979090463006,-pi,-pi) q[15];
cx q[13],q[15];
u(-7596.402425869207,0,0) q[15];
cx q[13],q[15];
u3(0.03138948908698556,0,0) q[15];
cx q[14],q[15];
u(-15192.804851738414,0,0) q[15];
cx q[14],q[15];
h q[14];
cp(-pi/2) q[13],q[14];
cp(-pi/4) q[12],q[14];
cp(-pi/8) q[11],q[14];
cp(-pi/16) q[10],q[14];
cp(-pi/32) q[9],q[14];
cp(-pi/64) q[8],q[14];
cp(-pi/128) q[7],q[14];
cp(-pi/256) q[6],q[14];
cp(-pi/512) q[5],q[14];
cp(-pi/1024) q[4],q[14];
cp(-pi/2048) q[3],q[14];
cp(-pi/4096) q[2],q[14];
cp(-pi/8192) q[1],q[14];
cp(-pi/16384) q[0],q[14];
h q[13];
cp(-pi/2) q[12],q[13];
cp(-pi/4) q[11],q[13];
cp(-pi/8) q[10],q[13];
cp(-pi/16) q[9],q[13];
cp(-pi/32) q[8],q[13];
cp(-pi/64) q[7],q[13];
cp(-pi/128) q[6],q[13];
cp(-pi/256) q[5],q[13];
cp(-pi/512) q[4],q[13];
cp(-pi/1024) q[3],q[13];
cp(-pi/2048) q[2],q[13];
cp(-pi/4096) q[1],q[13];
cp(-pi/8192) q[0],q[13];
h q[12];
cp(-pi/2) q[11],q[12];
cp(-pi/4) q[10],q[12];
cp(-pi/8) q[9],q[12];
cp(-pi/16) q[8],q[12];
cp(-pi/32) q[7],q[12];
cp(-pi/64) q[6],q[12];
cp(-pi/128) q[5],q[12];
cp(-pi/256) q[4],q[12];
cp(-pi/512) q[3],q[12];
cp(-pi/1024) q[2],q[12];
cp(-pi/2048) q[1],q[12];
cp(-pi/4096) q[0],q[12];
h q[11];
cp(-pi/2) q[10],q[11];
cp(-pi/4) q[9],q[11];
cp(-pi/8) q[8],q[11];
cp(-pi/16) q[7],q[11];
cp(-pi/32) q[6],q[11];
cp(-pi/64) q[5],q[11];
cp(-pi/128) q[4],q[11];
cp(-pi/256) q[3],q[11];
cp(-pi/512) q[2],q[11];
cp(-pi/1024) q[1],q[11];
cp(-pi/2048) q[0],q[11];
h q[10];
cp(-pi/2) q[9],q[10];
cp(-pi/4) q[8],q[10];
cp(-pi/8) q[7],q[10];
cp(-pi/16) q[6],q[10];
cp(-pi/32) q[5],q[10];
cp(-pi/64) q[4],q[10];
cp(-pi/128) q[3],q[10];
cp(-pi/256) q[2],q[10];
cp(-pi/512) q[1],q[10];
cp(-pi/1024) q[0],q[10];
h q[9];
cp(-pi/2) q[8],q[9];
cp(-pi/4) q[7],q[9];
cp(-pi/8) q[6],q[9];
cp(-pi/16) q[5],q[9];
cp(-pi/32) q[4],q[9];
cp(-pi/64) q[3],q[9];
cp(-pi/128) q[2],q[9];
cp(-pi/256) q[1],q[9];
cp(-pi/512) q[0],q[9];
h q[8];
cp(-pi/2) q[7],q[8];
cp(-pi/4) q[6],q[8];
cp(-pi/8) q[5],q[8];
cp(-pi/16) q[4],q[8];
cp(-pi/32) q[3],q[8];
cp(-pi/64) q[2],q[8];
cp(-pi/128) q[1],q[8];
cp(-pi/256) q[0],q[8];
h q[7];
cp(-pi/2) q[6],q[7];
cp(-pi/4) q[5],q[7];
cp(-pi/8) q[4],q[7];
cp(-pi/16) q[3],q[7];
cp(-pi/32) q[2],q[7];
cp(-pi/64) q[1],q[7];
cp(-pi/128) q[0],q[7];
h q[6];
cp(-pi/2) q[5],q[6];
cp(-pi/4) q[4],q[6];
cp(-pi/8) q[3],q[6];
cp(-pi/16) q[2],q[6];
cp(-pi/32) q[1],q[6];
cp(-pi/64) q[0],q[6];
h q[5];
cp(-pi/2) q[4],q[5];
cp(-pi/4) q[3],q[5];
cp(-pi/8) q[2],q[5];
cp(-pi/16) q[1],q[5];
cp(-pi/32) q[0],q[5];
h q[4];
cp(-pi/2) q[3],q[4];
cp(-pi/4) q[2],q[4];
cp(-pi/8) q[1],q[4];
cp(-pi/16) q[0],q[4];
h q[3];
cp(-pi/2) q[2],q[3];
cp(-pi/4) q[1],q[3];
cp(-pi/8) q[0],q[3];
h q[2];
cp(-pi/2) q[1],q[2];
cp(-pi/4) q[0],q[2];
h q[1];
cp(-pi/2) q[0],q[1];
h q[0];
u(15192.804851738414,0,0) q[15];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15];
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
