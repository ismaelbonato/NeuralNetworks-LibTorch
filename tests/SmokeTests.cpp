#include <iostream>

int main() {
    const int lhs = 2;
    const int rhs = 2;
    const int expected = 4;

    if ((lhs + rhs) != expected) {
        std::cerr << "Smoke test failed: arithmetic invariant broken" << '\n';
        return 1;
    }

    std::cout << "Smoke test passed" << '\n';
    return 0;
}
