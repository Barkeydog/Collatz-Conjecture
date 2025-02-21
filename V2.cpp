#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>   // for int64_t
#include <limits>
#include <optional>

// A small struct for rational numbers (to avoid floating-point issues).
// We'll store as numerator/denominator in 64-bit integers for demonstration.
// For more robust handling of large numbers, consider using an arbitrary-precision library.
struct Rational {
    int64_t num;
    int64_t den;

    Rational(int64_t n = 0, int64_t d = 1) : num(n), den(d) {
        normalize();
    }

    // Ensure gcd(num, den) = 1 and den>0
    void normalize() {
        if (den < 0) {
            den = -den;
            num = -num;
        }
        int64_t g = gcd(std::llabs(num), std::llabs(den));
        if (g > 1) {
            num /= g;
            den /= g;
        }
    }

    static int64_t gcd(int64_t a, int64_t b) {
        while (b != 0) {
            int64_t t = a % b;
            a = b;
            b = t;
        }
        return (a == 0) ? 1 : std::llabs(a);
    }
};

// Add two rationals
Rational operator+(const Rational& a, const Rational& b) {
    Rational res(a.num * b.den + b.num * a.den, a.den * b.den);
    res.normalize();
    return res;
}

// Multiply rational by an integer
Rational operator*(const Rational& r, int64_t k) {
    Rational res(r.num * k, r.den);
    res.normalize();
    return res;
}

// Divide rational by an integer
Rational operator/(const Rational& r, int64_t k) {
    Rational res(r.num, r.den * k);
    res.normalize();
    return res;
}

// Collatz step: if isOdd == true, do 3x + 1; else do x/2.
Rational collatzTransform(const Rational& x, bool isOdd) {
    if (isOdd) {
        // 3x + 1
        Rational threeX = x * 3;
        return threeX + Rational(1, 1);
    } else {
        // x / 2
        return x / 2;
    }
}

// Compute x_k as a function of x_0, given a pattern of odd/even over k steps.
// We'll return (A, B) such that x_k = A*x_0 + B.
std::pair<Rational, Rational> computeLinearForm(const std::vector<bool>& pattern) {
    // Start with x_0 = 1 * x_0 + 0
    // In rational form, that’s A=1, B=0 (both as rationals).
    Rational A(1, 1); 
    Rational B(0, 1);

    // For each step:
    //   x_{i+1} = 3 * x_i + 1      if odd
    //           = (1/2) * x_i      if even
    // If x_i = A*x_0 + B, then:
    //   If odd:  x_{i+1} = 3(A*x_0 + B) + 1 = 3A * x_0 + (3B + 1)
    //   If even: x_{i+1} = (A*x_0 + B)/2   = (A/2)*x_0 + B/2

    for (bool isOdd : pattern) {
        if (isOdd) {
            // (A, B) -> (3A, 3B+1)
            A = A * 3;
            B = B * 3 + Rational(1, 1);
        } else {
            // (A, B) -> (A/2, B/2)
            A = A / 2;
            B = B / 2;
        }
    }

    return {A, B};
}

// Try to solve for x0, given x_k = x_0 => A*x_0 + B = x_0 => (A - 1)*x_0 + B = 0.
// => x_0 = -B / (A - 1), if A != 1.
// Returns an empty optional if A=1 and B!=0 (no solution), or if the solution is not integer, etc.
std::optional<int64_t> solveForX0(const Rational& A, const Rational& B) {
    // We want A*x0 + B = x0 => (A - 1)*x0 + B = 0 => (A - 1)*x0 = -B.
    // x0 = -B / (A - 1).

    // Let A = aNum/aDen, B = bNum/bDen
    // A - 1 = (aNum/aDen) - 1 = (aNum - aDen) / aDen
    // So x0 = - (bNum/bDen) / ( (aNum - aDen)/aDen ) 
    //       = - (bNum/bDen) * (aDen/(aNum - aDen))
    //       = - bNum * aDen / [ bDen * (aNum - aDen) ]

    int64_t aNum = A.num;
    int64_t aDen = A.den;
    int64_t bNum = B.num;
    int64_t bDen = B.den;

    // A - 1 => numerator = (aNum - aDen), denominator = aDen
    int64_t topAminus1 = aNum - aDen;
    int64_t botAminus1 = aDen;

    // If A=1 => aNum= aDen => topAminus1=0
    if (topAminus1 == 0) {
        // Then A - 1 = 0, so the equation is 0*x0 + B=0 => B=0 => trivial or no solution
        if (bNum == 0) {
            // B=0 => x_k = x_0 for all x_0 => infinite solutions. Typically that means the identity transform.
            // But that indicates the pattern is effectively "do nothing," which can't happen with normal Collatz steps.
            return std::nullopt;
        } else {
            // No solution.
            return std::nullopt;
        }
    }

    // x0 numerator = -bNum * botAminus1
    // x0 denominator = bDen * topAminus1
    // We'll keep them in 64-bit for demonstration. We must watch for potential overflow though.
    int64_t x0num = -bNum * botAminus1;
    int64_t x0den = bDen * topAminus1;

    // We want x0 to be an integer => x0den must divide x0num evenly.
    // But let's reduce fraction first:
    auto gcd_val = Rational::gcd(std::llabs(x0num), std::llabs(x0den));
    x0num /= gcd_val;
    x0den /= gcd_val;

    if (x0den == 1 || x0den == -1) {
        // Then x0 is integer
        return x0num * (x0den); // if x0den = -1, multiply to get the correct sign
    } else {
        // Not integer
        return std::nullopt;
    }
}

// Given a valid integer x0, compute forward the sequence x0 -> x1 -> ... -> x_{k-1}
// according to the pattern (odd/even). Check that each x_i is indeed an integer
// with correct parity, and that x_k == x0.
bool verifyLoop(int64_t x0, const std::vector<bool>& pattern) {
    std::vector<int64_t> seq;
    seq.push_back(x0);

    for (bool isOdd : pattern) {
        int64_t current = seq.back();
        // Check parity matches the pattern
        if (isOdd) {
            if (current % 2 == 0) {
                return false; // mismatch
            }
            // next = 3*current + 1
            seq.push_back(3*current + 1);
        } else {
            if (current % 2 != 0) {
                return false; // mismatch
            }
            // next = current / 2
            if (current == 0) {
                // dividing 0 by 2 is 0, but that can't form a loop (it'll stay at 0).
                // We generally skip 0 in Collatz contexts.
                return false;
            }
            seq.push_back(current / 2);
        }
    }

    // Now check that seq[k] == seq[0].
    if (seq.back() == x0) {
        return true;
    } else {
        return false;
    }
}

int main() {
    // We'll search for cycles up to a certain length kMax.
    // Warning: the search grows as 2^k, so be cautious with large k.
    const int kMax = 12;

    // We will store any loops we find (besides the trivial 1->4->2->1).
    // Known small loop is [1,4,2], which we will likely rediscover.
    bool foundAny = false;

    for (int k = 2; k <= kMax; ++k) {
        // For each pattern of odd/even of length k, there are 2^k possibilities:
        int64_t patternsCount = (1LL << k);
        for (int64_t mask = 0; mask < patternsCount; ++mask) {
            // Build pattern vector of booleans:
            //    pattern[i] = true  => odd
            //    pattern[i] = false => even
            std::vector<bool> pattern(k);
            for (int i = 0; i < k; ++i) {
                bool bit = ( (mask >> i) & 1 ) == 1;
                pattern[i] = bit;
            }

            // Compute the linear form x_k = A*x_0 + B
            auto [A, B] = computeLinearForm(pattern);

            // Solve x_0 = -B / (A - 1) if possible
            auto maybeX0 = solveForX0(A, B);
            if (!maybeX0.has_value()) {
                continue;
            }
            int64_t x0 = maybeX0.value();

            // We want to skip negative or zero x0 for Collatz loops
            if (x0 <= 0) {
                continue;
            }

            // Verify that stepping forward actually matches pattern and returns to x0
            if (verifyLoop(x0, pattern)) {
                // We found a loop. Let's print it out:
                // But first, generate the sequence to show the loop.
                std::vector<int64_t> seq;
                seq.push_back(x0);
                for (bool isOdd : pattern) {
                    int64_t cur = seq.back();
                    int64_t nxt = (isOdd ? (3*cur + 1) : (cur/2));
                    seq.push_back(nxt);
                }
                // The last element should be x0 again.

                // Check if this is the trivial [1 -> 4 -> 2 -> 1] loop:
                // In ascending order, that loop is [1, 2, 4].
                // We'll do a quick check to avoid printing it repeatedly.
                // (Not a perfect check, but enough to skip the well-known loop.)
                // Let's gather unique elements in a set, see if it matches {1,2,4}.
                std::vector<int64_t> sortedLoop(seq.begin(), seq.end()-1); // exclude last repeated x0
                std::sort(sortedLoop.begin(), sortedLoop.end());
                sortedLoop.erase(std::unique(sortedLoop.begin(), sortedLoop.end()), sortedLoop.end());
                if (sortedLoop.size() == 3 &&
                    sortedLoop[0] == 1 && sortedLoop[1] == 2 && sortedLoop[2] == 4) {
                    continue; // skip printing the well-known loop
                }

                // Print the loop
                foundAny = true;
                std::cout << "Found loop of length " << k << ": ";
                for (int i = 0; i < (int)seq.size(); ++i) {
                    std::cout << seq[i];
                    if (i+1 < (int)seq.size()) std::cout << " -> ";
                }
                std::cout << " (pattern: ";
                for (bool b : pattern) {
                    std::cout << (b ? "odd" : "even") << " ";
                }
                std::cout << ")\n";
            }
        }
    }

    if (!foundAny) {
        std::cout << "No new loops found for k up to " << kMax << ".\n";
    }
    return 0;
}
