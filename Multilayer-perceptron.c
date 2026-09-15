#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// シグモイド関数
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// シグモイド関数の導関数 (微分)
// ※ 既に計算されたシグモイドの出力 y (= sigmoid(x)) を引数に取る
double sigmoid_derivative(double y) {
    return y * (1.0 - y);
}

int main() {
    // 1. 学習データ (XORの真理値表)
    double inputs[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    double targets[4][1] = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // 2. パラメータの設定
    double lr = 0.5;      // 学習率 (Learning Rate)
    int epochs = 20000;   // エポック数

    // 入力層 -> 隠れ層 の重みとバイアス (入力2, 隠れ2)
    double w_ih[2][2];
    double b_h[2];

    // 隠れ層 -> 出力層 の重みとバイアス (隠れ2, 出力1)
    double w_ho[2][1];
    double b_o[1];

    // 重みとバイアスの初期化 (簡単な擬似乱数)
    for (int i = 0; i < 2; i++) {
        b_h[i] = ((double)rand() / RAND_MAX) - 0.5;
        for (int j = 0; j < 2; j++) {
            w_ih[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }
    b_o[0] = ((double)rand() / RAND_MAX) - 0.5;
    for (int i = 0; i < 2; i++) {
        w_ho[i][0] = ((double)rand() / RAND_MAX) - 0.5;
    }

    // 3. 学習ループ (バックプロパゲーション)
    printf("=== 学習開始 ===\n");
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_error = 0.0;

        for (int p = 0; p < 4; p++) {
            // ----- A. 順伝播 (Forward Propagation) -----
            
            // 隠れ層の計算
            double hidden[2];
            for (int j = 0; j < 2; j++) {
                double sum = b_h[j];
                for (int i = 0; i < 2; i++) {
                    sum += inputs[p][i] * w_ih[i][j];
                }
                hidden[j] = sigmoid(sum);
            }

            // 出力層の計算
            double output[1];
            double sum_o = b_o[0];
            for (int j = 0; j < 2; j++) {
                sum_o += hidden[j] * w_ho[j][0];
            }
            output[0] = sigmoid(sum_o);

            // 2乗誤差の累積
            double err = targets[p][0] - output[0];
            total_error += 0.5 * err * err;

            // ----- B. 逆伝播 (Backpropagation) -----

            // 出力層のデルタ (勾配) 計算
            // δ_o = (Target - Output) * sigmoid'(Output)
            double delta_output = err * sigmoid_derivative(output[0]);

            // 隠れ層のデルタ計算
            // δ_h = (δ_o * w_ho) * sigmoid'(Hidden)
            double delta_hidden[2];
            for (int j = 0; j < 2; j++) {
                delta_hidden[j] = (delta_output * w_ho[j][0]) * sigmoid_derivative(hidden[j]);
            }

            // ----- C. 重みとバイアスの更新 -----

            // 隠れ層 -> 出力層の更新
            for (int j = 0; j < 2; j++) {
                w_ho[j][0] += lr * delta_output * hidden[j];
            }
            b_o[0] += lr * delta_output;

            // 入力層 -> 隠れ層の更新
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    w_ih[i][j] += lr * delta_hidden[j] * inputs[p][i];
                }
            }
        }

        // 5000エポックごとに誤差を表示
        if ((epoch + 1) % 5000 == 0) {
            printf("Epoch %5d - Total Loss: %.6f\n", epoch + 1, total_error);
        }
    }

    // 4. 学習後の検証
    printf("\n=== 検証結果 ===\n");
    for (int p = 0; p < 4; p++) {
        // 順伝播のみ実行して出力を確認
        double hidden[2];
        for (int j = 0; j < 2; j++) {
            double sum = b_h[j];
            for (int i = 0; i < 2; i++) {
                sum += inputs[p][i] * w_ih[i][j];
            }
            hidden[j] = sigmoid(sum);
        }

        double sum_o = b_o[0];
        for (int j = 0; j < 2; j++) {
            sum_o += hidden[j] * w_ho[j][0];
        }
        double out = sigmoid(sum_o);

        printf("入力: [%.0f, %.0f] -> 予測: %.4f (目標: %.0f) -> 二値分類結果: %d\n",
               inputs[p][0], inputs[p][1], out, targets[p][0], (out >= 0.5) ? 1 : 0);
    }

    return 0;
}
