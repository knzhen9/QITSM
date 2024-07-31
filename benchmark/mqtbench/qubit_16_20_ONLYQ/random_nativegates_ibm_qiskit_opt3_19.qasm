OPENQASM 2.0;
include "qelib1.inc";
qreg q[19];
creg meas[19];
rz(pi/2) q[0];
sx q[0];
rz(-3.120846425767195) q[0];
sx q[0];
rz(-1.4557906548997614) q[0];
rz(pi/2) q[1];
sx q[1];
rz(-2.8394793236697184) q[1];
sx q[1];
rz(6.072805557356929) q[1];
rz(1.7295709750491257) q[2];
sx q[2];
rz(-0.7362596955886556) q[2];
sx q[2];
rz(1.5808928260346349) q[2];
rz(pi/2) q[3];
sx q[3];
rz(2.7118568664763263) q[3];
sx q[3];
rz(-3.026586981694792) q[4];
rz(pi/2) q[5];
cx q[4],q[5];
x q[4];
rz(pi/4) q[5];
cx q[4],q[5];
rz(-0.6703924915024464) q[4];
sx q[4];
rz(-3.40825774599077) q[4];
rz(3*pi/4) q[5];
sx q[5];
rz(-0.933484913418873) q[5];
sx q[5];
cx q[1],q[5];
rz(-pi/4) q[5];
cx q[1],q[5];
rz(pi/4) q[5];
x q[6];
rz(-3.026586981694792) q[6];
rz(3.8173214781792772) q[8];
rz(2.318548159755715) q[9];
sx q[9];
rz(-pi/2) q[9];
rz(-2.948113975818636) q[10];
sx q[10];
cx q[9],q[10];
sx q[9];
rz(-0.5487328581253981) q[9];
sx q[9];
rz(0.5487328581253994) q[10];
cx q[9],q[10];
rz(-pi/2) q[9];
sx q[9];
rz(0.03764633043662968) q[9];
sx q[10];
rz(0.9869078942018428) q[10];
sx q[10];
rz(-1.0554533214255262) q[10];
cx q[10],q[2];
rz(-pi/4) q[2];
rz(-3*pi/2) q[11];
sx q[11];
rz(3*pi/4) q[11];
cx q[13],q[11];
rz(-pi/4) q[11];
cx q[7],q[11];
rz(pi/2) q[7];
sx q[7];
rz(4.606143296524289) q[7];
sx q[7];
rz(5*pi/2) q[7];
rz(pi/4) q[11];
cx q[13],q[11];
rz(pi/4) q[11];
sx q[11];
rz(pi) q[11];
rz(4.681331258267774) q[13];
sx q[14];
cx q[8],q[14];
sx q[8];
rz(3.2899582493108444) q[8];
sx q[8];
rz(7.630127886196097) q[8];
sx q[14];
rz(-pi) q[14];
cx q[0],q[14];
x q[0];
rz(0.46632287041293513) q[14];
cx q[0],q[14];
rz(0.11500567189500277) q[0];
sx q[0];
rz(-2.1832525826528304) q[0];
sx q[0];
rz(3*pi/4) q[0];
sx q[14];
cx q[14],q[2];
rz(pi/4) q[2];
cx q[10],q[2];
rz(-pi/4) q[2];
cx q[14],q[2];
rz(3*pi/4) q[2];
sx q[2];
rz(-pi/4) q[2];
cx q[14],q[10];
rz(-pi/4) q[10];
rz(pi/4) q[14];
cx q[14],q[10];
cx q[5],q[14];
cx q[14],q[5];
x q[5];
rz(-pi/2) q[5];
x q[15];
rz(pi/2) q[15];
cx q[6],q[15];
x q[6];
rz(pi/4) q[15];
cx q[6],q[15];
x q[6];
rz(pi/4) q[6];
rz(pi/4) q[15];
sx q[15];
cx q[11],q[15];
x q[11];
cx q[7],q[11];
cx q[11],q[7];
rz(-pi/2) q[15];
sx q[15];
rz(3.5573433667582526) q[15];
sx q[15];
rz(7*pi/2) q[15];
cx q[8],q[15];
x q[8];
rz(0.6710145274376558) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi) q[16];
sx q[16];
rz(0.5431690012775015) q[16];
sx q[16];
rz(-pi) q[17];
sx q[17];
rz(1.3327875459404757) q[17];
sx q[17];
cx q[12],q[17];
sx q[17];
rz(1.3327875459404757) q[17];
sx q[17];
rz(-pi) q[17];
cx q[12],q[17];
rz(-1.3713566267433344) q[12];
cx q[13],q[12];
rz(-0.36300324702506614) q[12];
sx q[12];
rz(-0.5103084980689179) q[12];
sx q[12];
cx q[13],q[12];
sx q[12];
rz(2.6312841555208752) q[12];
sx q[12];
rz(-1.6193542018733993) q[12];
x q[17];
rz(-3.075465274647229) q[17];
cx q[6],q[17];
x q[6];
rz(pi/4) q[17];
cx q[6],q[17];
rz(-0.20163993491231746) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[10],q[6];
rz(-pi/4) q[6];
cx q[2],q[6];
rz(pi/4) q[6];
cx q[10],q[6];
rz(-pi/4) q[6];
cx q[2],q[6];
rz(3*pi/4) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[10];
cx q[2],q[10];
rz(-pi/4) q[10];
cx q[2],q[10];
rz(pi/16) q[10];
x q[17];
rz(1.8326947201619834) q[17];
cx q[12],q[17];
x q[12];
rz(pi/4) q[17];
cx q[12],q[17];
rz(-1.1198740604782778) q[12];
cx q[12],q[7];
rz(-1.1213147578190659) q[7];
cx q[12],q[7];
rz(-2.020277895770727) q[7];
sx q[12];
rz(pi/2) q[12];
cx q[10],q[12];
rz(-pi/16) q[12];
cx q[10],q[12];
cx q[10],q[6];
rz(-pi/16) q[6];
rz(pi/16) q[12];
cx q[6],q[12];
rz(pi/16) q[12];
cx q[6],q[12];
cx q[10],q[6];
rz(9*pi/16) q[6];
rz(-pi/16) q[12];
cx q[6],q[12];
rz(-pi/16) q[12];
cx q[6],q[12];
cx q[6],q[14];
rz(pi/16) q[12];
rz(-pi/16) q[14];
cx q[14],q[12];
rz(pi/16) q[12];
cx q[14],q[12];
cx q[10],q[14];
rz(-pi/16) q[12];
rz(pi/16) q[14];
cx q[14],q[12];
rz(-pi/16) q[12];
cx q[14],q[12];
cx q[6],q[14];
sx q[6];
rz(2.1488849208215832) q[6];
rz(pi/16) q[12];
rz(-pi/16) q[14];
cx q[14],q[12];
rz(pi/16) q[12];
cx q[14],q[12];
cx q[10],q[14];
rz(-pi/16) q[12];
rz(pi/16) q[14];
cx q[14],q[12];
rz(-pi/16) q[12];
cx q[14],q[12];
rz(-1.7009403242670866) q[12];
sx q[12];
rz(-2.131118424142082) q[12];
sx q[12];
rz(-1.781430675616102) q[12];
cx q[12],q[15];
rz(-pi/16) q[15];
cx q[12],q[15];
rz(pi/16) q[15];
rz(-1.2427705544852454) q[17];
cx q[18],q[16];
sx q[16];
rz(0.543169001277501) q[16];
sx q[16];
rz(-pi) q[16];
cx q[18],q[16];
cx q[16],q[3];
sx q[3];
rz(2.7118568664763263) q[3];
sx q[3];
rz(-pi) q[3];
cx q[16],q[3];
rz(pi/4) q[3];
sx q[3];
cx q[13],q[16];
rz(-2.1200900473759106) q[16];
cx q[13],q[16];
cx q[1],q[13];
rz(1.2930005717400466) q[13];
sx q[13];
rz(-2.4141451621587446) q[13];
sx q[13];
cx q[1],q[13];
sx q[13];
rz(-2.4141451621587446) q[13];
sx q[13];
rz(2.791946771572742) q[13];
rz(2.1200900473759106) q[16];
x q[16];
cx q[16],q[8];
cx q[8],q[16];
rz(-0.02250294469789571) q[8];
cx q[8],q[14];
rz(-pi/4) q[14];
cx q[8],q[14];
rz(pi/2) q[14];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[12],q[16];
rz(-pi/16) q[16];
cx q[16],q[15];
rz(pi/16) q[15];
cx q[16],q[15];
cx q[12],q[16];
rz(-pi/16) q[15];
rz(pi/16) q[16];
cx q[16],q[15];
rz(-pi/16) q[15];
cx q[16],q[15];
rz(pi/16) q[15];
cx q[17],q[3];
x q[17];
cx q[17],q[0];
rz(-pi/4) q[0];
cx q[3],q[0];
rz(pi/4) q[0];
cx q[17],q[0];
rz(pi/4) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[6];
rz(-0.5780885940266869) q[6];
cx q[0],q[6];
x q[6];
cx q[13],q[17];
cx q[17],q[13];
cx q[13],q[17];
rz(pi/2) q[13];
sx q[13];
rz(4.033608981777413) q[13];
sx q[13];
rz(6.395402751701495) q[13];
rz(2.8555104872525194) q[17];
sx q[17];
rz(9.294381279725524) q[17];
sx q[17];
rz(7.855102701952067) q[17];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[4],q[18];
rz(2.3453197008450233) q[18];
cx q[4],q[18];
sx q[4];
rz(-0.7403987693557408) q[4];
cx q[9],q[18];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[4];
sx q[4];
rz(0.5978892499813746) q[4];
sx q[4];
rz(-pi) q[4];
sx q[9];
rz(0.5978892499813746) q[9];
sx q[9];
rz(-5*pi/2) q[9];
cx q[9],q[4];
rz(-2.4011938842340523) q[4];
cx q[1],q[4];
x q[1];
rz(pi/4) q[4];
cx q[1],q[4];
rz(2.6534171487122853) q[1];
sx q[1];
rz(-pi/2) q[1];
rz(-pi/4) q[4];
sx q[4];
rz(-pi) q[4];
sx q[9];
rz(5*pi/4) q[9];
cx q[9],q[10];
cx q[10],q[9];
cx q[9],q[10];
rz(-0.6719078465317854) q[9];
cx q[9],q[8];
rz(1.6536555507785957) q[8];
sx q[8];
rz(-1.7997235961246592) q[8];
sx q[8];
cx q[9],q[8];
sx q[8];
rz(-1.7997235961246592) q[8];
sx q[8];
rz(-0.8457544426832513) q[8];
rz(1.9778249211971906) q[10];
sx q[10];
rz(4.516293705437213) q[10];
sx q[10];
rz(14.46006323654041) q[10];
rz(pi/2) q[18];
sx q[18];
cx q[11],q[18];
rz(-3*pi/2) q[11];
sx q[11];
rz(3*pi/4) q[11];
cx q[2],q[11];
rz(-pi/4) q[11];
cx q[7],q[11];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[5];
cx q[5],q[7];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[7],q[4];
cx q[4],q[7];
cx q[7],q[4];
rz(-1.083270021503389) q[4];
rz(pi/4) q[11];
cx q[2],q[11];
sx q[2];
rz(0.6953534673916586) q[2];
sx q[2];
cx q[3],q[2];
sx q[2];
rz(0.6953534673916582) q[2];
sx q[2];
rz(-pi) q[2];
cx q[3],q[2];
cx q[2],q[0];
rz(pi/2) q[2];
sx q[2];
rz(2.105637648837221) q[2];
cx q[10],q[2];
rz(-0.5348413220423244) q[2];
cx q[10],q[2];
sx q[2];
rz(-1.3536872857340505) q[2];
sx q[2];
sx q[10];
rz(6.968670563844828) q[10];
sx q[10];
rz(5*pi/2) q[10];
rz(pi/4) q[11];
sx q[11];
rz(pi) q[11];
cx q[1],q[11];
cx q[11],q[1];
rz(-3.1205772510127643) q[1];
cx q[6],q[1];
sx q[1];
rz(1.0392613571312168) q[1];
sx q[1];
rz(-pi) q[1];
sx q[6];
rz(1.0392613571312168) q[6];
sx q[6];
rz(-5*pi/2) q[6];
cx q[6],q[1];
rz(3.1205772510127643) q[1];
sx q[6];
rz(-pi/2) q[6];
cx q[7],q[6];
cx q[6],q[7];
rz(-0.03846245497702738) q[6];
sx q[6];
rz(7.852568980831318) q[6];
rz(-pi/2) q[7];
rz(-pi/2) q[11];
cx q[9],q[11];
rz(-pi/16) q[11];
cx q[9],q[11];
cx q[9],q[0];
rz(-pi/16) q[0];
rz(pi/16) q[11];
cx q[0],q[11];
rz(pi/16) q[11];
cx q[0],q[11];
cx q[9],q[0];
rz(-0.7262172691087676) q[0];
rz(-pi/16) q[11];
cx q[0],q[11];
rz(-pi/16) q[11];
cx q[0],q[11];
cx q[0],q[8];
rz(-pi/16) q[8];
rz(pi/16) q[11];
cx q[8],q[11];
rz(pi/16) q[11];
cx q[8],q[11];
cx q[9],q[8];
rz(pi/16) q[8];
rz(-pi/16) q[11];
cx q[8],q[11];
rz(-pi/16) q[11];
cx q[8],q[11];
cx q[0],q[8];
cx q[0],q[7];
rz(-pi) q[7];
sx q[7];
rz(3*pi/4) q[7];
rz(-pi/16) q[8];
rz(pi/16) q[11];
cx q[8],q[11];
rz(pi/16) q[11];
cx q[8],q[11];
cx q[9],q[8];
rz(9*pi/16) q[8];
sx q[9];
rz(-pi/2) q[9];
rz(-pi/16) q[11];
cx q[8],q[11];
rz(-pi/16) q[11];
cx q[8],q[11];
rz(9*pi/16) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[2];
sx q[2];
rz(0.2171090410608456) q[2];
sx q[2];
rz(-pi) q[2];
cx q[11],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[6],q[2];
rz(pi/2) q[2];
sx q[2];
rz(5.481615189440238) q[2];
x q[11];
cx q[16],q[3];
rz(-pi/16) q[3];
cx q[3],q[15];
rz(pi/16) q[15];
cx q[3],q[15];
cx q[12],q[3];
rz(pi/16) q[3];
rz(-pi/16) q[15];
cx q[3],q[15];
rz(-pi/16) q[15];
cx q[3],q[15];
rz(pi/16) q[15];
cx q[16],q[3];
rz(-pi/16) q[3];
cx q[3],q[15];
rz(pi/16) q[15];
cx q[3],q[15];
cx q[12],q[3];
rz(pi/16) q[3];
sx q[12];
rz(3*pi/4) q[12];
rz(-pi/16) q[15];
cx q[3],q[15];
rz(-pi/16) q[15];
cx q[3],q[15];
rz(-7*pi/16) q[15];
sx q[15];
rz(-9*pi/4) q[15];
cx q[15],q[7];
rz(pi/4) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[0],q[7];
rz(pi/4) q[7];
sx q[16];
cx q[4],q[16];
sx q[4];
rz(-1.1206384545585788) q[4];
sx q[4];
rz(-2.1848524877944886) q[4];
sx q[16];
cx q[17],q[12];
rz(pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[1],q[12];
rz(pi/4) q[12];
cx q[3],q[12];
rz(-pi/4) q[12];
cx q[1],q[12];
x q[1];
rz(-3*pi/4) q[1];
cx q[1],q[16];
rz(pi/4) q[12];
cx q[3],q[12];
sx q[3];
cx q[8],q[3];
cx q[3],q[7];
rz(-pi/4) q[7];
cx q[0],q[7];
rz(pi/4) q[7];
cx q[3],q[7];
rz(pi/2) q[3];
rz(pi/4) q[7];
sx q[7];
rz(3*pi/4) q[7];
sx q[8];
rz(pi/4) q[12];
sx q[12];
rz(3*pi/4) q[12];
cx q[15],q[7];
rz(-pi/4) q[16];
cx q[1],q[16];
rz(-3*pi/4) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[3];
cx q[3],q[16];
rz(pi/2) q[3];
sx q[3];
rz(0.5085397834846677) q[3];
rz(-3*pi/2) q[16];
sx q[16];
rz(3*pi/4) q[16];
cx q[17],q[12];
rz(pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
sx q[17];
rz(3*pi/4) q[17];
cx q[12],q[17];
rz(-pi/4) q[17];
cx q[13],q[17];
sx q[13];
rz(3*pi/4) q[13];
cx q[10],q[13];
rz(-pi/4) q[13];
rz(pi/4) q[17];
cx q[12],q[17];
cx q[12],q[13];
sx q[12];
rz(-2.4035123541085435) q[12];
sx q[12];
rz(-pi/2) q[12];
cx q[8],q[12];
sx q[8];
rz(pi/4) q[8];
rz(pi/2) q[12];
sx q[12];
rz(-2.4035123541085426) q[12];
sx q[12];
rz(pi/4) q[13];
cx q[10],q[13];
cx q[10],q[4];
sx q[4];
rz(0.9442467850123482) q[4];
sx q[4];
rz(-pi) q[4];
cx q[10],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/4) q[13];
sx q[13];
rz(-2.5883439910130335) q[13];
cx q[13],q[11];
rz(-2.1240449893716558) q[11];
cx q[13],q[11];
rz(2.1240449893716558) q[11];
cx q[11],q[4];
sx q[11];
sx q[13];
rz(-2.4035123541085435) q[13];
sx q[13];
rz(-pi/2) q[13];
cx q[15],q[8];
rz(pi/4) q[8];
cx q[12],q[8];
rz(-pi/4) q[8];
rz(pi/4) q[12];
cx q[15],q[8];
rz(3*pi/4) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[15],q[12];
rz(-pi/4) q[12];
cx q[15],q[12];
cx q[8],q[12];
rz(-2.4035123541085435) q[12];
sx q[12];
rz(-pi/2) q[12];
cx q[9],q[12];
sx q[9];
rz(-pi/2) q[12];
sx q[12];
rz(-0.7380802994812505) q[12];
sx q[12];
rz(pi/4) q[12];
cx q[0],q[12];
rz(pi/4) q[12];
cx q[9],q[12];
rz(pi/4) q[9];
rz(-pi/4) q[12];
cx q[0],q[12];
cx q[0],q[9];
rz(-pi/4) q[9];
cx q[0],q[9];
sx q[0];
rz(-pi/2) q[0];
rz(3*pi/4) q[12];
sx q[12];
rz(pi) q[12];
cx q[12],q[9];
cx q[12],q[11];
x q[11];
rz(-3.026586981694792) q[11];
x q[12];
cx q[12],q[16];
x q[12];
rz(-3.0265869816947912) q[12];
sx q[15];
rz(pi/4) q[15];
rz(-pi/4) q[16];
rz(-pi/4) q[17];
sx q[17];
rz(0.2107614217932987) q[17];
sx q[17];
rz(-pi) q[18];
sx q[18];
rz(-2.605435367836332) q[18];
cx q[5],q[18];
rz(-2.1069536125483586) q[18];
cx q[5],q[18];
cx q[14],q[5];
cx q[5],q[14];
rz(-pi) q[5];
sx q[5];
cx q[5],q[7];
rz(pi/4) q[7];
sx q[7];
rz(pi/2) q[7];
rz(1.881086824807972) q[14];
sx q[14];
rz(4.53699571164481) q[14];
sx q[14];
rz(8.757793836141165) q[14];
cx q[14],q[7];
rz(pi/4) q[7];
cx q[6],q[7];
rz(-pi/4) q[7];
cx q[14],q[7];
rz(pi/4) q[7];
cx q[6],q[7];
sx q[6];
rz(-3.000264312744087) q[6];
rz(pi/4) q[7];
sx q[7];
rz(3*pi/4) q[7];
cx q[5],q[7];
rz(pi/2) q[5];
cx q[5],q[10];
rz(pi/4) q[7];
sx q[7];
rz(-pi/2) q[7];
cx q[7],q[6];
rz(-1.712124667640603) q[6];
cx q[7],q[6];
rz(-pi/2) q[6];
sx q[6];
rz(-pi/2) q[6];
rz(-pi/4) q[10];
cx q[5],q[10];
cx q[5],q[4];
rz(-pi/4) q[4];
rz(1.3297959511513922) q[10];
rz(pi/2) q[18];
sx q[18];
rz(3*pi/4) q[18];
cx q[18],q[17];
sx q[17];
rz(1.7815577485881953) q[17];
sx q[17];
rz(-pi) q[17];
cx q[18],q[17];
rz(-pi) q[17];
sx q[17];
rz(3*pi/4) q[17];
cx q[1],q[17];
cx q[1],q[13];
sx q[1];
rz(pi/4) q[1];
rz(pi/2) q[13];
sx q[13];
rz(-2.4035123541085426) q[13];
sx q[13];
cx q[14],q[1];
rz(pi/4) q[1];
cx q[13],q[1];
rz(-pi/4) q[1];
rz(pi/4) q[13];
cx q[14],q[1];
rz(3*pi/4) q[1];
sx q[1];
rz(4.393761377431478) q[1];
cx q[14],q[13];
rz(-pi/4) q[13];
cx q[14],q[13];
cx q[1],q[13];
sx q[1];
rz(-1.874376650454077) q[1];
sx q[1];
rz(2.4053406101782473) q[1];
cx q[6],q[14];
cx q[13],q[15];
rz(4.922132731873485) q[14];
cx q[6],q[14];
sx q[6];
rz(2.023460641617662) q[6];
rz(-1.9646863136009145) q[14];
sx q[14];
rz(-1.2347509522136324) q[14];
sx q[14];
rz(-0.1616906647136691) q[14];
rz(-pi/4) q[15];
cx q[7],q[15];
sx q[7];
rz(pi) q[7];
rz(pi/4) q[15];
cx q[13],q[15];
sx q[13];
rz(pi) q[13];
cx q[13],q[7];
sx q[13];
rz(-pi/2) q[13];
rz(-3*pi/4) q[15];
sx q[15];
rz(1.7539668921805491) q[15];
cx q[11],q[15];
x q[11];
rz(pi/4) q[15];
cx q[11],q[15];
rz(1.0967533761418116) q[11];
rz(2.841106160139553) q[15];
rz(-0.3627863721135345) q[17];
sx q[17];
cx q[3],q[17];
sx q[17];
rz(1.9934081180788104) q[17];
sx q[17];
rz(-pi) q[17];
cx q[3],q[17];
sx q[3];
rz(-2.9281304623065276) q[3];
sx q[3];
rz(-0.6703844893459632) q[3];
cx q[17],q[4];
rz(pi/4) q[4];
cx q[5],q[4];
rz(-pi/4) q[4];
cx q[17],q[4];
rz(3*pi/4) q[4];
sx q[4];
rz(2.459404438096345) q[4];
cx q[10],q[4];
rz(-0.888608111301449) q[4];
cx q[10],q[4];
x q[4];
rz(pi/2) q[4];
cx q[3],q[4];
x q[3];
rz(1.4861404455928668) q[4];
cx q[3],q[4];
rz(-1.3711347736978645) q[3];
sx q[3];
rz(-0.623502069346026) q[3];
sx q[3];
rz(1.8923370468374276) q[3];
rz(-2.868907739967329) q[4];
cx q[11],q[10];
rz(0.3950453764267081) q[10];
cx q[14],q[10];
sx q[10];
rz(0.07733822069175433) q[10];
sx q[10];
rz(-pi) q[10];
sx q[14];
rz(0.07733822069175433) q[14];
sx q[14];
rz(-5*pi/2) q[14];
cx q[14],q[10];
rz(1.1757509503681884) q[10];
sx q[14];
rz(-pi/2) q[14];
cx q[15],q[14];
rz(-2.6579355947539014) q[14];
cx q[15],q[14];
rz(-2.0544533856307887) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[17],q[5];
rz(-pi/4) q[5];
rz(-pi/4) q[17];
cx q[17],q[5];
cx q[5],q[16];
rz(pi/2) q[5];
sx q[5];
rz(pi) q[5];
rz(pi/4) q[16];
sx q[16];
rz(2.809311545918093) q[16];
sx q[16];
rz(-pi/2) q[16];
cx q[12],q[16];
x q[12];
rz(0.8135094393021951) q[16];
cx q[12],q[16];
rz(-1.2164619749922316) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[16];
sx q[16];
rz(-1.3380420737903567) q[16];
sx q[16];
rz(1.8101250067025614) q[16];
cx q[6],q[16];
x q[6];
rz(pi/4) q[16];
cx q[6],q[16];
x q[6];
rz(-1.7264135683062594) q[6];
x q[16];
rz(-3*pi/4) q[16];
sx q[17];
sx q[18];
rz(pi/2) q[18];
cx q[2],q[18];
rz(-pi/4) q[18];
cx q[2],q[18];
sx q[2];
rz(-2.5903670531236784) q[2];
sx q[2];
rz(-1.0317015666944442) q[2];
cx q[1],q[2];
sx q[1];
rz(-0.4599573202965366) q[1];
sx q[1];
rz(0.8233735224777098) q[2];
cx q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(-1.2306655685741035) q[1];
sx q[1];
rz(2.710944369516783) q[1];
rz(-0.6826773397186692) q[2];
sx q[2];
rz(-1.4363301414131664) q[2];
sx q[2];
rz(-5.075218587184742) q[2];
cx q[2],q[4];
sx q[2];
rz(-pi/2) q[2];
rz(-pi) q[4];
x q[4];
cx q[17],q[1];
sx q[1];
rz(2.0405427105572898) q[1];
sx q[1];
rz(-pi) q[1];
sx q[17];
rz(2.0405427105572898) q[17];
sx q[17];
cx q[17],q[1];
rz(2.4762977060670543) q[1];
cx q[1],q[7];
rz(-pi/4) q[7];
cx q[1],q[7];
rz(3*pi/4) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[10],q[7];
rz(1.4485743637511421) q[7];
cx q[10],q[7];
rz(pi/2) q[7];
sx q[7];
sx q[17];
rz(-pi/4) q[17];
cx q[17],q[3];
rz(-pi/4) q[3];
rz(3*pi/4) q[18];
sx q[18];
rz(-1.5998893840316488) q[18];
cx q[18],q[8];
rz(-2.3110446166077585) q[8];
cx q[18],q[8];
rz(0.740248289812862) q[8];
sx q[8];
cx q[0],q[8];
sx q[0];
rz(-1.4825801182156002) q[0];
sx q[0];
rz(1.4825801182156015) q[8];
cx q[0],q[8];
rz(1.5324198631909054) q[0];
sx q[0];
rz(-1.749560433319873) q[0];
sx q[0];
rz(1.9456536416663042) q[0];
sx q[8];
rz(pi/2) q[8];
cx q[18],q[9];
rz(-0.016056816347834554) q[9];
cx q[18],q[9];
rz(0.8014549797452828) q[9];
cx q[18],q[8];
rz(-pi/4) q[8];
cx q[9],q[8];
rz(pi/4) q[8];
cx q[18],q[8];
rz(-pi/4) q[8];
cx q[9],q[8];
x q[8];
rz(3*pi/4) q[8];
cx q[8],q[0];
rz(-pi) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[9],q[18];
cx q[11],q[0];
rz(-pi/16) q[0];
cx q[11],q[0];
rz(pi/16) q[0];
cx q[11],q[1];
rz(-pi/16) q[1];
cx q[1],q[0];
rz(pi/16) q[0];
cx q[1],q[0];
rz(-pi/16) q[0];
cx q[11],q[1];
rz(pi/16) q[1];
cx q[1],q[0];
rz(-pi/16) q[0];
cx q[1],q[0];
rz(pi/16) q[0];
cx q[1],q[8];
rz(-pi/16) q[8];
cx q[8],q[0];
rz(pi/16) q[0];
cx q[8],q[0];
rz(-pi/16) q[0];
cx q[11],q[8];
rz(pi/16) q[8];
cx q[8],q[0];
rz(-pi/16) q[0];
cx q[8],q[0];
rz(pi/16) q[0];
cx q[1],q[8];
x q[1];
rz(pi/2) q[1];
rz(-pi/16) q[8];
cx q[8],q[0];
rz(pi/16) q[0];
cx q[8],q[0];
rz(-pi/16) q[0];
cx q[11],q[8];
rz(pi/16) q[8];
cx q[8],q[0];
rz(-pi/16) q[0];
cx q[8],q[0];
rz(9*pi/16) q[0];
sx q[0];
rz(-pi/2) q[0];
cx q[8],q[14];
sx q[11];
rz(pi/2) q[11];
cx q[6],q[11];
rz(5.888026210943938) q[11];
cx q[6],q[11];
sx q[6];
rz(-1.6448506385823434) q[6];
sx q[6];
rz(-2.1288408859823735) q[6];
rz(pi/2) q[11];
sx q[11];
rz(-1.523478462878698) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[4],q[11];
x q[4];
rz(pi/2) q[11];
sx q[11];
rz(-1.6181141907110943) q[11];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/4) q[18];
cx q[9],q[18];
cx q[13],q[9];
rz(5.548968590502123) q[9];
cx q[13],q[9];
rz(0.8315802044639007) q[9];
sx q[9];
rz(-2.058841601856468) q[9];
sx q[9];
rz(-2.1146733154764803) q[9];
sx q[13];
rz(1.6433249674837533) q[13];
rz(0.0031083678672225563) q[18];
cx q[5],q[18];
rz(5.664166523141638) q[18];
cx q[5],q[18];
cx q[5],q[3];
rz(pi/4) q[3];
cx q[17],q[3];
rz(-pi/4) q[3];
cx q[5],q[3];
rz(3*pi/4) q[3];
sx q[3];
rz(3*pi/4) q[3];
cx q[5],q[17];
cx q[7],q[3];
rz(pi/4) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[0],q[3];
rz(pi/4) q[3];
cx q[9],q[3];
rz(-pi/4) q[3];
cx q[0],q[3];
sx q[0];
cx q[0],q[2];
sx q[0];
rz(-pi/2) q[2];
sx q[2];
rz(-0.7380802994812505) q[2];
sx q[2];
rz(pi/4) q[2];
rz(pi/4) q[3];
cx q[9],q[3];
rz(pi/4) q[3];
sx q[3];
rz(3*pi/4) q[3];
cx q[7],q[3];
rz(-pi/4) q[3];
sx q[3];
rz(pi/2) q[3];
sx q[7];
rz(0.334572733595186) q[7];
sx q[7];
cx q[14],q[9];
cx q[9],q[14];
cx q[14],q[9];
rz(-pi/4) q[17];
cx q[5],q[17];
cx q[5],q[16];
rz(-pi/4) q[16];
cx q[5],q[16];
sx q[5];
rz(-0.832716027313646) q[5];
sx q[5];
rz(-pi/2) q[5];
cx q[4],q[5];
rz(-pi) q[4];
sx q[4];
rz(3*pi/4) q[4];
rz(pi/2) q[5];
sx q[5];
rz(-0.832716027313646) q[5];
rz(pi/4) q[16];
cx q[13],q[16];
cx q[16],q[13];
cx q[13],q[16];
cx q[9],q[13];
cx q[13],q[9];
cx q[9],q[13];
rz(2.071631389253273) q[9];
rz(0.812263044849352) q[13];
rz(3*pi/4) q[16];
sx q[16];
rz(-pi/2) q[16];
rz(0.8461567087467522) q[17];
sx q[17];
rz(6.595472699805312) q[17];
sx q[17];
rz(14.05115052843561) q[17];
cx q[8],q[17];
cx q[17],q[8];
rz(-pi/2) q[17];
cx q[14],q[17];
cx q[14],q[11];
cx q[11],q[14];
rz(pi/2) q[11];
sx q[17];
rz(0.8449991545613731) q[17];
sx q[17];
rz(-0.3484046458180927) q[17];
cx q[17],q[4];
rz(-pi/4) q[4];
cx q[18],q[12];
cx q[10],q[18];
sx q[12];
rz(-pi) q[12];
cx q[1],q[12];
rz(5.832011141632526) q[12];
cx q[1],q[12];
sx q[1];
rz(pi/4) q[1];
cx q[8],q[1];
rz(-pi/4) q[1];
cx q[3],q[1];
rz(pi/4) q[1];
cx q[8],q[1];
rz(pi/4) q[1];
sx q[1];
rz(-1.3018936429225167) q[1];
rz(-1.1732194119734616) q[8];
sx q[8];
rz(-0.7525283438221493) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[7],q[8];
sx q[7];
rz(-1.212028689476824) q[7];
sx q[7];
rz(1.3198923230425348) q[8];
cx q[7],q[8];
rz(pi/4) q[7];
sx q[7];
rz(-pi/2) q[7];
rz(-2.1857682922977677) q[8];
sx q[8];
rz(-2.0048312757159783) q[8];
sx q[8];
rz(-0.041576792144159214) q[8];
rz(pi/2) q[12];
sx q[12];
rz(-pi) q[12];
cx q[18],q[10];
cx q[10],q[18];
rz(-pi) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[15];
cx q[15],q[10];
rz(pi/2) q[10];
sx q[10];
rz(-1.905797337594871) q[10];
sx q[10];
cx q[10],q[11];
cx q[11],q[10];
rz(pi/2) q[11];
sx q[11];
rz(3*pi/4) q[11];
rz(pi/2) q[15];
sx q[15];
rz(2.3871550588646056) q[15];
cx q[15],q[12];
rz(-0.816358732069709) q[12];
cx q[15],q[12];
rz(2.3871550588646056) q[12];
cx q[12],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
sx q[12];
rz(2.3357777482592947) q[12];
cx q[13],q[15];
cx q[14],q[3];
rz(5.335462994130735) q[3];
cx q[14],q[3];
rz(pi/2) q[3];
sx q[3];
rz(3.8495310393864868) q[3];
rz(-1.5976612082468002) q[15];
cx q[13],q[15];
rz(-1.5439314453429933) q[15];
sx q[15];
rz(0.9615340702664747) q[15];
sx q[15];
cx q[18],q[2];
rz(pi/4) q[2];
cx q[0],q[2];
rz(pi/4) q[0];
rz(-pi/4) q[2];
cx q[18],q[2];
rz(3*pi/4) q[2];
sx q[2];
rz(pi/4) q[2];
cx q[18],q[0];
rz(-pi/4) q[0];
rz(1.5366396329493934) q[18];
cx q[18],q[0];
cx q[2],q[0];
rz(-1.0648232770178592) q[0];
sx q[2];
rz(-pi) q[2];
cx q[2],q[16];
sx q[2];
rz(-pi/4) q[2];
sx q[2];
cx q[6],q[0];
rz(1.831793812670968) q[0];
sx q[0];
rz(-2.8827817193364442) q[0];
sx q[0];
cx q[6],q[0];
rz(-2.426809783309386) q[0];
sx q[0];
rz(-1.2039073134712979) q[0];
sx q[0];
rz(0.739189212806826) q[0];
sx q[6];
rz(-pi/2) q[6];
cx q[6],q[1];
sx q[1];
rz(2.3090057980179512) q[1];
sx q[1];
rz(-pi) q[1];
sx q[6];
rz(2.3090057980179512) q[6];
sx q[6];
cx q[6],q[1];
rz(2.8726899697174133) q[1];
cx q[1],q[12];
rz(pi/2) q[6];
sx q[6];
rz(1.8117350146465068) q[6];
cx q[6],q[14];
rz(-0.7649814214643981) q[12];
cx q[1],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[0],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[13],q[1];
cx q[1],q[13];
cx q[13],q[1];
rz(-1.2263062516638719) q[14];
cx q[6],q[14];
sx q[6];
rz(pi/2) q[6];
rz(3.9493400404180488) q[14];
cx q[14],q[12];
rz(2.443596646876557) q[12];
cx q[14],q[12];
rz(pi/2) q[12];
sx q[12];
rz(1.4278765524222874) q[16];
cx q[2],q[16];
rz(-pi) q[2];
sx q[2];
cx q[2],q[15];
sx q[15];
rz(0.9615340702664747) q[15];
sx q[15];
rz(-pi) q[15];
cx q[2],q[15];
rz(2.21879255497351) q[2];
cx q[15],q[4];
rz(pi/4) q[4];
rz(2.464442836067414) q[15];
cx q[1],q[15];
rz(-2.464442836067414) q[15];
cx q[1],q[15];
rz(2.9203800013044026) q[1];
rz(-pi/2) q[16];
sx q[16];
rz(-pi) q[16];
cx q[16],q[9];
cx q[9],q[16];
cx q[16],q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[3],q[9];
sx q[3];
rz(-3.0946568697908567) q[3];
sx q[3];
rz(1.6482561043956503) q[3];
rz(pi/2) q[9];
sx q[9];
rz(pi/4) q[9];
rz(-pi/2) q[16];
cx q[11],q[16];
cx q[3],q[11];
rz(-pi/4) q[11];
cx q[3],q[11];
rz(5.907623867709416) q[11];
rz(pi/2) q[16];
sx q[16];
rz(pi) q[16];
cx q[16],q[6];
rz(1.9164065388485196) q[6];
cx q[16],q[6];
rz(pi/2) q[6];
sx q[6];
rz(-3*pi/4) q[6];
cx q[6],q[9];
rz(-pi/4) q[9];
cx q[14],q[9];
rz(pi/4) q[9];
cx q[6],q[9];
rz(-pi/4) q[9];
cx q[14],q[9];
rz(3*pi/4) q[9];
sx q[9];
rz(-1.8145057676083187) q[9];
cx q[8],q[9];
rz(-2.8978832127763714) q[9];
cx q[8],q[9];
rz(pi/2) q[9];
sx q[9];
rz(2.8857883350265467) q[9];
cx q[14],q[6];
rz(-pi/4) q[6];
cx q[14],q[6];
cx q[14],q[3];
rz(-0.7920892907790313) q[3];
cx q[14],q[3];
rz(3.4017277474812726) q[3];
cx q[3],q[8];
rz(2.719452877036872) q[8];
sx q[16];
rz(-pi) q[16];
cx q[17],q[4];
cx q[0],q[17];
rz(pi/4) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[2],q[4];
cx q[4],q[2];
cx q[4],q[1];
rz(-2.9203800013044026) q[1];
cx q[4],q[1];
sx q[1];
rz(-pi/2) q[4];
sx q[4];
rz(-1.2784985436188254) q[4];
cx q[12],q[1];
x q[12];
rz(2.422705551809608) q[12];
cx q[14],q[12];
rz(-0.10416053122880431) q[12];
sx q[12];
rz(-2.5712163861940738) q[12];
sx q[12];
cx q[14],q[12];
sx q[12];
rz(-2.5712163861940747) q[12];
sx q[12];
rz(-2.318545020580803) q[12];
sx q[14];
rz(-pi/2) q[14];
cx q[14],q[8];
sx q[8];
rz(1.2024733398526397) q[8];
sx q[8];
rz(-pi) q[8];
sx q[14];
rz(1.2024733398526397) q[14];
sx q[14];
cx q[14],q[8];
rz(-1.1486565502419754) q[8];
sx q[8];
rz(-pi) q[8];
rz(pi/2) q[14];
sx q[14];
rz(0.6157983762316963) q[14];
cx q[15],q[2];
cx q[2],q[15];
cx q[15],q[2];
rz(-3*pi/2) q[2];
sx q[2];
rz(3*pi/4) q[2];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[6],q[15];
rz(4.607707280570853) q[15];
cx q[6],q[15];
rz(-pi/2) q[15];
sx q[15];
rz(pi) q[15];
rz(-1.3963996158053775) q[17];
sx q[17];
rz(-3.031282111062522) q[17];
sx q[17];
cx q[0],q[17];
sx q[0];
rz(-1.5317210620868522) q[0];
sx q[0];
rz(-3.026586981694792) q[0];
sx q[17];
rz(-3.031282111062522) q[17];
sx q[17];
rz(3.168268322815774) q[17];
cx q[17],q[16];
rz(-2.3334507214271136) q[16];
cx q[17],q[16];
rz(-2.747966552546761) q[16];
sx q[16];
rz(-2.825991371153788) q[16];
cx q[6],q[16];
rz(0.9405521302930957) q[6];
rz(-6.634355272117626) q[16];
cx q[16],q[12];
rz(-2.7904226886517534) q[12];
cx q[16],q[12];
rz(4.36121901544665) q[12];
sx q[16];
rz(3*pi/4) q[16];
cx q[15],q[16];
rz(pi/4) q[16];
sx q[16];
rz(pi/2) q[16];
sx q[17];
rz(-1.0203232691562576) q[17];
sx q[17];
rz(2.5493083881665033) q[17];
sx q[18];
rz(-pi/4) q[18];
cx q[5],q[18];
rz(-pi/4) q[18];
cx q[10],q[18];
cx q[13],q[10];
rz(pi/4) q[18];
cx q[5],q[18];
sx q[5];
rz(1.89002045538151) q[5];
sx q[5];
rz(0.06274907050322653) q[5];
cx q[5],q[13];
rz(-pi/4) q[13];
cx q[5],q[13];
cx q[5],q[2];
rz(-pi/4) q[2];
cx q[7],q[2];
cx q[1],q[7];
rz(pi/4) q[2];
cx q[5],q[2];
rz(pi/4) q[2];
sx q[2];
rz(-1.4557906548998947) q[2];
cx q[7],q[1];
sx q[1];
rz(0.7674752124770351) q[7];
sx q[7];
rz(7.925656618919779) q[7];
sx q[7];
rz(8.657302748292345) q[7];
cx q[7],q[3];
rz(4.192685979458471) q[3];
cx q[7],q[3];
rz(1.1010387894660578) q[3];
sx q[3];
rz(-0.8039480469117315) q[3];
x q[13];
rz(-pi/2) q[13];
cx q[13],q[17];
rz(-0.5922842654232898) q[17];
sx q[17];
rz(-1.0203232691562576) q[17];
sx q[17];
rz(0.6608096665600875) q[17];
cx q[2],q[17];
rz(pi/4) q[17];
cx q[9],q[17];
rz(-pi/4) q[17];
cx q[2],q[17];
rz(pi/4) q[17];
cx q[9],q[17];
sx q[9];
rz(5.24799764163666) q[9];
sx q[9];
rz(3*pi/2) q[9];
rz(pi/4) q[17];
sx q[17];
rz(3*pi/4) q[17];
cx q[13],q[17];
cx q[13],q[16];
rz(pi/4) q[16];
cx q[12],q[16];
rz(-pi/4) q[16];
cx q[13],q[16];
rz(pi/4) q[16];
cx q[12],q[16];
sx q[12];
rz(pi/2) q[12];
cx q[13],q[12];
rz(pi/2) q[12];
sx q[12];
rz(-0.7822480148429563) q[12];
rz(pi/4) q[16];
sx q[16];
rz(3*pi/4) q[16];
cx q[15],q[16];
rz(pi/4) q[16];
sx q[16];
rz(1.2979046445879752) q[16];
rz(pi/4) q[17];
sx q[17];
rz(-0.5045137362003738) q[17];
cx q[2],q[17];
x q[2];
rz(pi/4) q[17];
cx q[2],q[17];
x q[2];
rz(3.4186425687415714) q[2];
rz(0.1643556921661382) q[17];
sx q[17];
rz(-1.4363164186055188) q[17];
sx q[17];
rz(0.047059956722073915) q[17];
rz(3*pi/4) q[18];
sx q[18];
rz(1.742429468565339) q[18];
cx q[10],q[18];
rz(-0.17163314177044273) q[18];
cx q[10],q[18];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[11],q[10];
rz(6.050906344027643) q[10];
cx q[11],q[10];
cx q[4],q[10];
rz(3.00730984372108) q[10];
cx q[4],q[10];
sx q[4];
rz(-0.6905379604433755) q[4];
sx q[4];
rz(-1.681434754711935) q[4];
cx q[4],q[8];
sx q[4];
rz(-1.0250309819707581) q[4];
sx q[4];
rz(1.5104211128677911) q[8];
cx q[4],q[8];
rz(2.8030555682418443) q[4];
sx q[4];
rz(-1.4767180994173223) q[4];
sx q[4];
rz(3.1085293787701556) q[4];
sx q[8];
rz(-pi/2) q[8];
rz(pi/2) q[10];
sx q[10];
rz(-pi) q[10];
cx q[10],q[1];
rz(pi/2) q[1];
sx q[1];
rz(-1.4557906548998956) q[1];
x q[10];
rz(1.4438233459982865) q[10];
cx q[11],q[5];
rz(-2.4189435606955714) q[5];
cx q[11],q[5];
cx q[11],q[6];
rz(0.9314680044209336) q[6];
sx q[6];
rz(-0.9190353139875) q[6];
sx q[6];
cx q[11],q[6];
cx q[2],q[11];
sx q[6];
rz(-0.9190353139875) q[6];
sx q[6];
rz(0.021014092491927983) q[6];
cx q[9],q[6];
sx q[6];
rz(2.6729764211305227) q[6];
sx q[6];
rz(-pi) q[6];
sx q[9];
rz(2.6729764211305236) q[9];
sx q[9];
cx q[9],q[6];
rz(-1.893034227205959) q[6];
rz(pi/2) q[9];
sx q[9];
rz(-0.2569965504272833) q[9];
cx q[1],q[9];
x q[1];
rz(pi/4) q[9];
cx q[1],q[9];
rz(-0.6703924915024455) q[1];
sx q[1];
rz(-pi/2) q[1];
x q[9];
rz(-1.3137997763676132) q[9];
rz(-2.206327981816449) q[11];
cx q[2],q[11];
sx q[2];
rz(pi/2) q[2];
rz(-0.7816015652907531) q[11];
cx q[11],q[15];
cx q[13],q[1];
rz(-pi/4) q[1];
cx q[13],q[1];
rz(3*pi/4) q[1];
sx q[1];
rz(1.6026929582348077) q[1];
sx q[13];
rz(pi/2) q[13];
cx q[14],q[10];
rz(0.9549979505632002) q[10];
sx q[10];
rz(-1.9737983849393679) q[10];
sx q[10];
cx q[14],q[10];
sx q[10];
rz(-1.9737983849393679) q[10];
sx q[10];
rz(-1.4464049997080966) q[10];
cx q[10],q[7];
rz(3.3267340383086825) q[7];
cx q[10],q[7];
rz(0.003274070674319063) q[7];
rz(-pi/4) q[15];
cx q[11],q[15];
rz(5*pi/16) q[15];
cx q[16],q[6];
rz(-1.2979046445879752) q[6];
cx q[16],q[6];
rz(2.8687009713828715) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[15],q[6];
rz(-pi/16) q[6];
cx q[15],q[6];
rz(pi/16) q[6];
sx q[16];
cx q[17],q[10];
rz(1.6534733447059793) q[10];
sx q[10];
rz(-2.2579491469238295) q[10];
sx q[10];
cx q[17],q[10];
rz(-1.210671148380419) q[10];
sx q[10];
rz(-1.1651608942132476) q[10];
sx q[10];
rz(-1.5942076902632758) q[10];
cx q[17],q[2];
rz(pi/2) q[2];
sx q[2];
rz(-pi) q[2];
rz(3*pi/4) q[18];
sx q[18];
rz(-1.0000517870900385) q[18];
sx q[18];
rz(-0.9577805721963566) q[18];
cx q[18],q[5];
rz(-2.4891913351782162) q[5];
cx q[18],q[5];
rz(-0.6524013184115769) q[5];
sx q[5];
rz(-pi) q[5];
cx q[14],q[5];
rz(2.8981904842219564) q[5];
sx q[5];
rz(-pi/2) q[5];
cx q[5],q[16];
sx q[5];
rz(-0.031985797613926614) q[5];
sx q[5];
cx q[15],q[14];
rz(-pi/16) q[14];
cx q[14],q[6];
rz(pi/16) q[6];
cx q[14],q[6];
rz(-pi/16) q[6];
cx q[15],q[14];
rz(pi/16) q[14];
cx q[14],q[6];
rz(-pi/16) q[6];
cx q[14],q[6];
rz(pi/16) q[6];
cx q[14],q[8];
rz(-pi/16) q[8];
cx q[8],q[6];
rz(pi/16) q[6];
cx q[8],q[6];
rz(-pi/16) q[6];
cx q[15],q[8];
rz(pi/16) q[8];
cx q[8],q[6];
rz(-pi/16) q[6];
cx q[8],q[6];
rz(pi/16) q[6];
cx q[14],q[8];
rz(-pi/16) q[8];
cx q[8],q[6];
rz(pi/16) q[6];
cx q[8],q[6];
rz(-pi/16) q[6];
sx q[14];
rz(pi) q[14];
cx q[15],q[8];
rz(-15*pi/16) q[8];
cx q[8],q[6];
rz(-pi/16) q[6];
cx q[8],q[6];
rz(9*pi/16) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[3];
cx q[3],q[6];
cx q[6],q[3];
rz(-pi/2) q[3];
sx q[3];
rz(5.130331147289391) q[3];
sx q[3];
rz(6.686207957363057) q[3];
sx q[8];
cx q[10],q[6];
rz(-pi/4) q[6];
sx q[15];
rz(0.34087060426750426) q[15];
sx q[15];
rz(-3.7317060128075488) q[15];
cx q[15],q[17];
rz(0.03198579761392812) q[16];
cx q[5],q[16];
rz(pi/2) q[5];
sx q[5];
rz(-1.0288003327652646) q[5];
sx q[5];
rz(-pi/2) q[5];
cx q[1],q[5];
sx q[1];
rz(pi/2) q[5];
cx q[8],q[5];
rz(2.311085463959472) q[5];
sx q[5];
rz(-1.2024618339342723) q[5];
sx q[5];
rz(-1.1985722864638682) q[5];
rz(-pi/2) q[16];
sx q[16];
cx q[14],q[16];
rz(1.331682425112953) q[16];
cx q[14],q[16];
sx q[14];
rz(-1.3256986219219753) q[14];
cx q[14],q[6];
rz(pi/4) q[6];
cx q[10],q[6];
rz(-pi/4) q[6];
cx q[14],q[6];
x q[6];
rz(-2.3382282476158247) q[6];
cx q[14],q[10];
rz(-pi/4) q[10];
cx q[14],q[10];
cx q[10],q[2];
rz(-pi/4) q[2];
sx q[14];
rz(-1.0404684773064723) q[14];
sx q[14];
rz(-2.4410277630583153) q[14];
cx q[6],q[14];
sx q[6];
rz(-pi/2) q[6];
rz(-pi/2) q[14];
sx q[14];
rz(-0.7380802994812505) q[14];
sx q[14];
rz(1.0612189070855482) q[14];
cx q[1],q[14];
sx q[1];
rz(2.143217673690865) q[1];
sx q[1];
sx q[14];
rz(2.143217673690865) q[14];
sx q[14];
rz(-pi) q[14];
cx q[1],q[14];
rz(-3*pi/2) q[1];
sx q[1];
rz(0.790850898489655) q[1];
rz(-1.0612189070855487) q[14];
sx q[14];
rz(3.316257897198275) q[14];
rz(-pi) q[16];
sx q[16];
rz(-2.2516615085515364) q[16];
rz(2.2315297011606674) q[17];
sx q[17];
rz(-1.6135633983941116) q[17];
sx q[17];
cx q[15],q[17];
sx q[15];
rz(4.055086201173993) q[15];
sx q[15];
rz(7.870459936891118) q[15];
cx q[15],q[8];
cx q[8],q[15];
rz(-3*pi/2) q[8];
sx q[8];
rz(-1.3778810349793922) q[8];
rz(-pi/2) q[15];
sx q[15];
rz(-pi/2) q[15];
cx q[15],q[14];
sx q[14];
rz(3.1116097825532876) q[14];
sx q[14];
rz(-pi) q[14];
sx q[15];
rz(3.1116097825532876) q[15];
sx q[15];
cx q[15],q[14];
rz(-1.7454615704033776) q[14];
rz(-3*pi/2) q[15];
sx q[15];
rz(-pi/2) q[15];
sx q[17];
rz(-1.6135633983941116) q[17];
sx q[17];
rz(2.14182538868421) q[17];
sx q[18];
rz(0.9934360172816641) q[18];
cx q[0],q[18];
x q[0];
rz(pi/4) q[18];
cx q[0],q[18];
x q[0];
rz(2.241188818297342) q[0];
cx q[9],q[0];
rz(-pi/4) q[0];
cx q[9],q[0];
rz(3*pi/4) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[9],q[13];
cx q[12],q[0];
rz(pi/2) q[0];
sx q[0];
rz(-2.6261941196606013) q[0];
sx q[12];
rz(pi/2) q[12];
rz(1.5299985069170676) q[13];
cx q[9],q[13];
x q[9];
rz(0.4372186843460404) q[9];
rz(-pi/2) q[13];
sx q[13];
rz(-pi) q[13];
x q[18];
rz(3.1086435890094943) q[18];
cx q[7],q[18];
x q[7];
rz(pi/4) q[18];
cx q[7],q[18];
x q[7];
rz(5.880992552108436) q[7];
cx q[7],q[4];
rz(2.5003975968937953) q[4];
cx q[7],q[4];
rz(pi/2) q[4];
sx q[4];
rz(1.0910183634069943) q[4];
cx q[4],q[12];
sx q[4];
rz(-1.8651621888188208) q[4];
sx q[4];
rz(-2.0228178534477435) q[4];
cx q[4],q[6];
sx q[4];
rz(-pi/2) q[6];
sx q[6];
rz(-0.7380802994812505) q[6];
sx q[6];
rz(pi/4) q[6];
cx q[7],q[0];
rz(-0.38647947900061785) q[0];
sx q[0];
rz(-1.5289615353999082) q[0];
sx q[0];
cx q[7],q[0];
sx q[0];
rz(-1.5289615353999082) q[0];
sx q[0];
rz(-0.9509584690488104) q[0];
rz(-pi/2) q[12];
sx q[12];
rz(-0.9186755361331898) q[12];
sx q[12];
rz(2.5400564533316086) q[12];
cx q[17],q[0];
rz(-0.8245613946286223) q[0];
sx q[0];
rz(-2.5894526627280516) q[0];
sx q[0];
cx q[17],q[0];
sx q[0];
rz(-2.5894526627280516) q[0];
sx q[0];
rz(1.1205835578179641) q[0];
cx q[0],q[12];
rz(-1.0447790758640025) q[12];
cx q[0],q[12];
cx q[0],q[3];
cx q[3],q[0];
cx q[0],q[3];
rz(4.483764475230662) q[0];
rz(-1.8337975686704748) q[3];
sx q[3];
rz(5.049062840338128) q[3];
sx q[3];
rz(9.544548123376199) q[3];
rz(1.0447790758640025) q[12];
rz(2.5741317599536844) q[18];
sx q[18];
rz(-0.3506649992855806) q[18];
sx q[18];
rz(-1.9094273403674) q[18];
cx q[18],q[11];
rz(-1.894277765044837) q[11];
sx q[11];
rz(-1.4608491388977543) q[11];
sx q[11];
cx q[18],q[11];
sx q[11];
rz(1.680743514692038) q[11];
sx q[11];
rz(-1.8744608427878715) q[11];
cx q[11],q[13];
x q[11];
rz(pi/4) q[13];
cx q[11],q[13];
rz(-0.02161767559077621) q[11];
sx q[11];
rz(-1.969711711542793) q[11];
sx q[11];
cx q[11],q[6];
rz(pi/4) q[6];
cx q[4],q[6];
rz(pi/4) q[4];
rz(-pi/4) q[6];
cx q[11],q[6];
rz(3*pi/4) q[6];
sx q[6];
rz(8.319182071062038) q[6];
cx q[11],q[4];
rz(-pi/4) q[4];
rz(pi/4) q[11];
cx q[11],q[4];
cx q[6],q[4];
x q[4];
rz(-1.0347729161837318) q[4];
cx q[9],q[4];
rz(-0.948121010681044) q[4];
sx q[4];
rz(-1.3418322356044765) q[4];
sx q[4];
cx q[9],q[4];
sx q[4];
rz(-1.3418322356044765) q[4];
sx q[4];
rz(-1.1586987267250173) q[4];
rz(1.2053290169354436) q[13];
cx q[13],q[16];
rz(-0.8899311450382572) q[16];
cx q[13],q[16];
rz(pi/2) q[16];
sx q[16];
rz(3*pi/4) q[16];
sx q[18];
cx q[7],q[18];
sx q[7];
rz(-pi/2) q[7];
cx q[17],q[7];
rz(-pi/4) q[7];
cx q[2],q[7];
rz(pi/4) q[7];
cx q[17],q[7];
rz(-pi/4) q[7];
cx q[2],q[7];
cx q[2],q[17];
rz(3*pi/4) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[12];
rz(6.223383322037425) q[12];
cx q[7],q[12];
rz(-pi/4) q[17];
cx q[2],q[17];
sx q[2];
rz(-1.5745291566015114) q[18];
cx q[18],q[10];
rz(-0.8289831975070322) q[10];
cx q[18],q[10];
rz(6.1296518552140355) q[10];
cx q[10],q[17];
cx q[10],q[13];
rz(-2.1590760041172103) q[13];
sx q[13];
rz(-2.2488771699906227) q[13];
sx q[13];
cx q[10],q[13];
cx q[10],q[11];
cx q[11],q[10];
sx q[13];
rz(-2.2488771699906227) q[13];
sx q[13];
rz(1.7391451505792146) q[13];
cx q[13],q[3];
rz(pi/4) q[3];
sx q[3];
rz(pi/2) q[3];
rz(9*pi/16) q[17];
sx q[18];
rz(-pi/2) q[18];
cx q[2],q[18];
sx q[2];
rz(-pi/2) q[18];
sx q[18];
rz(-0.7380802994812505) q[18];
sx q[18];
rz(pi/4) q[18];
cx q[16],q[18];
rz(pi/4) q[18];
cx q[2],q[18];
rz(pi/4) q[2];
rz(-pi/4) q[18];
cx q[16],q[18];
cx q[16],q[2];
rz(-pi/4) q[2];
cx q[16],q[2];
cx q[0],q[16];
rz(-2.9129681484357652) q[16];
cx q[0],q[16];
sx q[0];
rz(pi/2) q[0];
rz(2.9129681484357652) q[16];
cx q[16],q[3];
rz(pi/4) q[3];
rz(3*pi/4) q[18];
sx q[18];
rz(pi) q[18];
cx q[18],q[2];
rz(-pi) q[2];
sx q[2];
rz(3*pi/4) q[2];
cx q[5],q[2];
rz(pi/4) q[2];
sx q[2];
cx q[5],q[3];
rz(-pi/4) q[3];
cx q[16],q[3];
rz(pi/4) q[3];
cx q[5],q[3];
rz(pi/4) q[3];
sx q[3];
rz(3*pi/4) q[3];
cx q[13],q[3];
rz(pi/4) q[3];
sx q[3];
rz(pi/2) q[3];
sx q[18];
rz(pi/2) q[18];
cx q[17],q[18];
rz(-pi/16) q[18];
cx q[17],q[18];
cx q[17],q[7];
rz(-pi/16) q[7];
rz(pi/16) q[18];
cx q[7],q[18];
rz(pi/16) q[18];
cx q[7],q[18];
cx q[17],q[7];
rz(3.3379421944391554) q[7];
rz(-pi/16) q[18];
cx q[7],q[18];
rz(-pi/16) q[18];
cx q[7],q[18];
cx q[7],q[12];
rz(-pi/16) q[12];
rz(pi/16) q[18];
cx q[12],q[18];
rz(pi/16) q[18];
cx q[12],q[18];
cx q[17],q[12];
rz(pi/16) q[12];
rz(-pi/16) q[18];
cx q[12],q[18];
rz(-pi/16) q[18];
cx q[12],q[18];
cx q[7],q[12];
x q[7];
rz(-pi/16) q[12];
rz(pi/16) q[18];
cx q[12],q[18];
rz(pi/16) q[18];
cx q[12],q[18];
cx q[17],q[12];
rz(5*pi/16) q[12];
sx q[17];
rz(pi/2) q[17];
cx q[2],q[17];
rz(-pi/4) q[17];
cx q[6],q[17];
rz(pi/4) q[17];
cx q[2],q[17];
rz(pi/4) q[2];
rz(-pi/4) q[17];
cx q[6],q[17];
cx q[6],q[2];
rz(-pi/4) q[2];
cx q[6],q[2];
rz(3*pi/4) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-pi/16) q[18];
cx q[12],q[18];
rz(-pi/16) q[18];
cx q[12],q[18];
cx q[12],q[0];
rz(-pi/4) q[0];
cx q[12],q[0];
rz(3*pi/4) q[0];
sx q[0];
rz(pi/2) q[0];
x q[18];
rz(-3.3379421944391545) q[18];
cx q[18],q[8];
sx q[8];
rz(2.6850213018922826) q[8];
sx q[8];
rz(-pi) q[8];
sx q[18];
rz(2.6850213018922826) q[18];
sx q[18];
cx q[18],q[8];
rz(2.9486773617742887) q[8];
rz(-3*pi/2) q[18];
sx q[18];
rz(-pi/2) q[18];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18];
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
