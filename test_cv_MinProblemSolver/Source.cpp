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
		cout << "Error: 実行ファイルと同じディレクトリにcsvデータファイルを用意してください．\n"
			<< "Formatは以下の通りです．\n"
			<< "2行目までラベル行\n"
			<< "3行目以降:(ラベル列),(0番目の光源)R,G,B,(1番目の光源)R,G,B,...,(19番目の光源)R,G,B\n"
			<< "3行目から24行のデータを読み込む" << endl;
		exit(0);
	}
	Mat m(rows, cols, CV_64FC3);

	string str;	//	行
	getline(csvFile, str);
	getline(csvFile, str);	//	2行目まで読み飛ばす
	//	3行目以降rows行分
	for (auto i = 0; i < rows; i++) {
		getline(csvFile, str);
		string token;	//	セル
		istringstream stream(str);
		vector<double> data;
		//	行をカンマ分割して数値データとして格納
		getline(stream, token, ',');
		for (auto j = 0; j < cols * 3; j++) {
			getline(stream, token, ',');
			double temp = stod(token);
			data.push_back(temp);
		}
		//	数値データを行列にコピー
		for (auto j = 0; j < cols; j++) {
			m.at<Vec3d>(i, j) = Vec3d(data[3 * j], data[3 * j + 1], data[3 * j + 2]);
		}
	}
	//	読み込んだデータを表示
	cout << m << endl;
	cout << "-------------------------------------------------\n" << endl;

	return m;
}

//	最小化ソルバーの目的関数
class CostFunction : public MinProblemSolver::Function {
public:
	double calc(const double* x) const {
		//	csvデータの分解
		Mat projColors = csvdata.row(0).clone();
		Mat patchColors = csvdata(Rect(0, 19, csvdata.cols, 6)).clone();
		//	パラメータベクトルの解釈
		Vec3d gamma_p(x[0], x[1], x[2]);
		Vec3d gamma_c(x[3], x[4], x[5]);
		Vec3d offset(x[6], x[7], x[8]);
		Vec3d cam_th(x[9], x[10], x[11]);
		Mat cvtMatPro2Cam = (Mat_<double>(3, 3) <<
			x[12], x[13], x[14],
			x[15], x[16], x[17],
			x[18], x[19], x[20]);

		//	全ての投影色に対して誤差ベクトルを求める
		vector<Vec3d> error;
		for (int j = 0; j < csvdata.cols; j++) {
			//	Pro階調値 > 正規化した輝度
			auto P = projColors.at<Vec3d>(j);
			auto Lp = Vec3d(
				pow(P[0] / 255.0, gamma_p[0]),
				pow(P[1] / 255.0, gamma_p[1]),
				pow(P[2] / 255.0, gamma_p[2])
			);
			//	正規化した輝度 > Cam線形色空間での入射光
			Mat cp_m = cvtMatPro2Cam * Mat(Lp);
			auto cp = (Vec3d)cp_m + offset;
			//	Cam線形色空間での入射光 > 反射光
			for (auto i = 0; i < 6; i++) {
				double r = 0.9 * gray[i] / gray[0];
				auto cc = r * cp;
				//	反射光 > Cam階調値
				auto C_est = 255.0 * Vec3d(
					pow(max(0.0, cc[0] - cam_th[0]), 1.0 / gamma_c[0]),
					pow(max(0.0, cc[1] - cam_th[1]), 1.0 / gamma_c[1]),
					pow(max(0.0, cc[2] - cam_th[2]), 1.0 / gamma_c[2])
				);
				//	実測値との誤差を記録
				auto C_msr = patchColors.at<Vec3d>(i, j);
				error.push_back(C_est - C_msr);
			}
		}
		//	誤差2乗和を計算
		double cost = 0.0;
		for (auto e : error) {
			cost += e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
		}

		return cost;
	}
	virtual int getDims() const { return 21; }		//	パラメータベクトルの次元数
};

//	OpenCVの最小化問題ソルバーの一つ，滑降シンプレックス法（ネルダー・ミード法）を利用した最適化
void test_downhill()
{
	Mat param = (Mat_<double>(1, 21) <<
		2., 2., 2.,			//	プロジェクタガンマ
		1., 1., 1.,			//	カメラガンマ
		0.01, 0.01, 0.01,	//	環境光
		0.01, 0.01, 0.01,	//	カメラ感度
		0.6, 0.0, 0.0,		//	ProCam色変換行列
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
	//	data読込
	//	0行目がPro階調値，1-24行目が各パッチの反射色のCam階調値(最小化には 0, 19-24 を使う)
	//	列は投影色の違い
	//	RGB順であることに注意
	csvdata = readCSV("./test.csv", 25, 20);
	test_downhill();
}