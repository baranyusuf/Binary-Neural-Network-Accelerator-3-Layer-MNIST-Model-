#include "bnn.h"
#include "golden.h"
#include <iostream>
#include <string>

using namespace std;

template <size_t N>
void verify_layer(const string& name, const int golden[N], const int predicted[N]) {
    cout << "-----------------------------------------------------------" << endl;
    cout << "Verifying " << name << endl;

    bool PASS = true;
    for (size_t i = 0; i < N; i++) {
        if (golden[i] != predicted[i]) {
            if (PASS) cout << "Mismatches:\n";
            PASS = false;
            cout << "  idx " << i << " -> expected " << golden[i]
                 << ", got " << predicted[i] << endl;
        }
    }

    if (PASS)
        cout << name << " PASSED." << endl;
    else
        cout << name << " FAILED." << endl;
}

int main() {
    cout << "===========================================================" << endl;
    cout << "                BNN Layer-by-Layer Test                    " << endl;
    cout << "===========================================================" << endl;

    for (int sample_idx = 0; sample_idx < 10; sample_idx++) {
        cout << "\n===== SAMPLE " << sample_idx << " =====" << endl;

        // Layer 1
        int layer1_out[128] = {0};
        layer1(INPUT_BITS[sample_idx], layer1_out);
        verify_layer<128>("Layer 1 Output", LAYER1_OUTPUT[sample_idx], layer1_out);

        // Layer 2
        int layer2_out[64] = {0};
        layer2(layer1_out, layer2_out);
        verify_layer<64>("Layer 2 Output", LAYER2_OUTPUT[sample_idx], layer2_out);

        // Layer 3
        int layer3_out[10] = {0};
        layer3(layer2_out, layer3_out);
        verify_layer<10>("Layer 3 Output", LAYER3_OUTPUT[sample_idx], layer3_out);

        // Final BNN Output
        int final_out[10] = {0};
        bnn(INPUT_BITS[sample_idx], final_out);
        verify_layer<10>("Final BNN Output", OUTPUT[sample_idx], final_out);
    }

    cout << "\nAll tests completed." << endl;
    return 0;
}
