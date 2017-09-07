#include <opencv2\opencv.hpp>
#include <fstream>
#include <string>
#include <sstream>
#include <array>

using namespace cv;
using namespace std;

Mat csvdata;
array<double, 6> gray = { 90.01, 59.1, 36.2, 19.77, 9.0, 3.13 };

Mat readCSV(const char* filename, const int rows, const int cols)
{
	ifstream csvFile(filename);
	if (!csvFile) {
		cout << "Error: ���s�t�@�C���Ɠ����f�B���N�g����csv�f�[�^�t�@�C����p�ӂ��Ă��������D\n"
			<< "Format�͈ȉ��̒ʂ�ł��D\n"
			<< "2�s�ڂ܂Ń��x���s\n"
			<< "3�s�ڈȍ~:(���x����),(0�Ԗڂ̌���)R,G,B,(1�Ԗڂ̌���)R,G,B,...,(19�Ԗڂ̌���)R,G,B\n"
			<< "3�s�ڂ���24�s�̃f�[�^��ǂݍ���" << endl;
		exit(0);
	}
	Mat m(rows, cols, CV_64FC3);

	string str;	//	�s
	getline(csvFile, str);
	getline(csvFile, str);	//	2�s�ڂ܂œǂݔ�΂�
	//	3�s�ڈȍ~rows�s��
	for (auto i = 0; i < rows; i++) {
		getline(csvFile, str);
		string token;	//	�Z��
		istringstream stream(str);
		vector<double> data;
		//	�s���J���}�������Đ��l�f�[�^�Ƃ��Ċi�[
		getline(stream, token, ',');
		for (auto j = 0; j < cols * 3; j++) {
			getline(stream, token, ',');
			double temp = stod(token);
			data.push_back(temp);
		}
		//	���l�f�[�^���s��ɃR�s�[
		for (auto j = 0; j < cols; j++) {
			m.at<Vec3d>(i, j) = Vec3d(data[3 * j], data[3 * j + 1], data[3 * j + 2]);
		}
	}
	//	�ǂݍ��񂾃f�[�^��\��
	cout << m << endl;
	cout << "-------------------------------------------------\n" << endl;

	return m;
}

//	�ŏ����\���o�[�̖ړI�֐�
class CostFunction : public MinProblemSolver::Function {
public:
	double calc(const double* x) const {
		//	csv�f�[�^�̕���
		Mat projColors = csvdata.row(0).clone();
		Mat patchColors = csvdata(Rect(0, 19, csvdata.cols, 6)).clone();
		//	�p�����[�^�x�N�g���̉���
		Vec3d gamma_p(x[0], x[1], x[2]);
		Vec3d gamma_c(x[3], x[4], x[5]);
		Vec3d offset(x[6], x[7], x[8]);
		Vec3d cam_th(x[9], x[10], x[11]);
		Mat cvtMatPro2Cam = (Mat_<double>(3, 3) <<
			x[12], x[13], x[14],
			x[15], x[16], x[17],
			x[18], x[19], x[20]);

		//	�S�Ă̓��e�F�ɑ΂��Č덷�x�N�g�������߂�
		vector<Vec3d> error;
		for (int j = 0; j < csvdata.cols; j++) {
			//	Pro�K���l > ���K�������P�x
			auto P = projColors.at<Vec3d>(j);
			auto Lp = Vec3d(
				pow(P[0] / 255.0, gamma_p[0]),
				pow(P[1] / 255.0, gamma_p[1]),
				pow(P[2] / 255.0, gamma_p[2])
			);
			//	���K�������P�x > Cam���`�F��Ԃł̓��ˌ�
			Mat cp_m = cvtMatPro2Cam * Mat(Lp);
			auto cp = (Vec3d)cp_m + offset;
			//	Cam���`�F��Ԃł̓��ˌ� > ���ˌ�
			for (auto i = 0; i < 6; i++) {
				double r = 0.9 * gray[i] / gray[0];
				auto cc = r * cp;
				//	���ˌ� > Cam�K���l
				auto C_est = 255.0 * Vec3d(
					pow(max(0.0, cc[0] - cam_th[0]), 1.0 / gamma_c[0]),
					pow(max(0.0, cc[1] - cam_th[1]), 1.0 / gamma_c[1]),
					pow(max(0.0, cc[2] - cam_th[2]), 1.0 / gamma_c[2])
				);
				//	�����l�Ƃ̌덷���L�^
				auto C_msr = patchColors.at<Vec3d>(i, j);
				error.push_back(C_est - C_msr);
			}
		}
		//	�덷2��a���v�Z
		double cost = 0.0;
		for (auto e : error) {
			cost += e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
		}

		return cost;
	}
	virtual int getDims() const { return 21; }		//	�p�����[�^�x�N�g���̎�����
};

//	OpenCV�̍ŏ������\���o�[�̈�C���~�V���v���b�N�X�@�i�l���_�[�E�~�[�h�@�j�𗘗p�����œK��
void test_downhill()
{
	Mat param = (Mat_<double>(1, 21) <<
		2., 2., 2.,			//	�v���W�F�N�^�K���}
		1., 1., 1.,			//	�J�����K���}
		0.01, 0.01, 0.01,	//	����
		0.01, 0.01, 0.01,	//	�J�������x
		0.6, 0.0, 0.0,		//	ProCam�F�ϊ��s��
		0.0, 0.6, 0.0,
		0.0, 0.0, 0.6);
	Mat step = (Mat_<double>(1, 21) << 
		0.5, 0.5, 0.5,
		0.5, 0.5, 0.5,
		0.5, 0.5, 0.5,
		0.5, 0.5, 0.5,
		0.5, 0.5, 0.5,
		0.5, 0.5, 0.5,
		0.5, 0.5, 0.5);

	auto ptr_func(new CostFunction());
	auto solver = DownhillSolver::create();
	solver->setFunction(ptr_func);
	solver->setInitStep(step);

	double res = solver->minimize(param);

	cout << "res: " << res << endl;
	cout << "params:\n" << param << endl;
}

void main() 
{
	//	data�Ǎ�
	//	0�s�ڂ�Pro�K���l�C1-24�s�ڂ��e�p�b�`�̔��ːF��Cam�K���l(�ŏ����ɂ� 0, 19-24 ���g��)
	//	��͓��e�F�̈Ⴂ
	//	RGB���ł��邱�Ƃɒ���
	csvdata = readCSV("./test.csv", 25, 20);
	test_downhill();
}