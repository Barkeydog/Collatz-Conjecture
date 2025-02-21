#include <iostream>
#include <vector>
#include <algorithm>
#include <gmpxx.h>

// A small class for rational numbers using GMP's mpz_class for numerator & denominator.
class BigRational {
public:
    mpz_class num;  // Numerator
    mpz_class den;  // Denominator

    // Constructors
    BigRational() : num(0), den(1) {}
    BigRational(const mpz_class &n, const mpz_class &d) : num(n), den(d) {
        normalize();
    }
    BigRational(long long v) : num(v), den(1) {
    }

    // Normalization: ensure gcd(num, den)=1, and den>0
    void normalize() {
        if (den < 0) {
            den = -den;
            num = -num;
        }
        mpz_class g = gcd(mpz_class_abs(num), mpz_class_abs(den));
        if (g != 0) {
            num /= g;
            den /= g;
        }
    }

    // Utility to get absolute value as mpz_class
    static mpz_class mpz_class_abs(const mpz_class &x) {
        return (x < 0) ? -x : x;
    }

    // gcd for mpz_class
    static mpz_class gcd(const mpz_class &a, const mpz_class &b) {
        mpz_class g;
        mpz_gcd(g.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
        return g;
    }
};

// Add two BigRationals
BigRational operator+(const BigRational &a, const BigRational &b) {
    // a/b + c/d = (ad + bc) / bd
    BigRational res(a.num * b.den + b.num * a.den, a.den * b.den);
    res.normalize();
    return res;
}

// Multiply BigRational by an integer
BigRational operator*(const BigRational &r, long long k) {
    BigRational res(r.num * k, r.den);
    res.normalize();
    return res;
}

// Multiply two BigRationals
BigRational operator*(const BigRational &a, const BigRational &b) {
    BigRational res(a.num * b.num, a.den * b.den);
    res.normalize();
    return res;
}

// Divide BigRational by an integer
BigRational operator/(const BigRational &r, long long k) {
    BigRational res(r.num, r.den * k);
    res.normalize();
    return res;
}

// Divide BigRational by BigRational
BigRational operator/(const BigRational &a, const BigRational &b) {
    // a/b / c/d = (a/b) * (d/c) = ad / bc
    BigRational res(a.num * b.den, a.den * b.num);
    res.normalize();
    return res;
}

// Collatz step in BigRational form: if isOdd == true, do 3x + 1; else do x/2.
BigRational collatzTransform(const BigRational &x, bool isOdd) {
    if (isOdd) {
        // 3x + 1
        BigRational threeX = x * 3;
        return threeX + BigRational(1);
    } else {
        // x/2
        return x / 2;
    }
}

// Compute x_k as a function of x_0, given a pattern of odd/even over k steps.
// Returns (A, B) such that x_k = A*x_0 + B (both BigRational).
std::pair<BigRational, BigRational> computeLinearForm(const std::vector<bool> &pattern) {
    // Start with x_0 = 1 * x_0 + 0
    BigRational A(1), B(0);

    // For each step:
    //   x_{i+1} = 3 * x_i + 1      if odd
    //           = (1/2)*x_i        if even
    // If x_i = A*x_0 + B, then:
    //   If odd:  x_{i+1} = 3(A*x_0 + B) + 1 = 3A*x_0 + (3B + 1)
    //   If even: x_{i+1} = (A*x_0 + B)/2   = (A/2)*x_0 + B/2
    for (bool isOdd : pattern) {
        if (isOdd) {
            // (A,B) -> (3A, 3B + 1)
            A = A * 3;
            B = B * 3 + BigRational(1);
        } else {
            // (A,B) -> (A/2, B/2)
            A = A / 2;
            B = B / 2;
        }
    }
    return {A, B};
}

// Attempt to solve x_0 from the equation x_k = x_0 => A*x_0 + B = x_0 => (A - 1)*x_0 + B=0
// => x_0 = -B / (A - 1), if A != 1. Return mpz_class if integer solution, else "empty".
std::optional<mpz_class> solveForX0(const BigRational &A, const BigRational &B) {
    // If A=1 => A.num/A.den=1 => A.num= A.den => then topAminus1=0
    // We'll compute A-1 = (A.num - A.den)/A.den
    mpz_class topAminus1 = A.num - A.den;  // numerator of (A-1)
    mpz_class botAminus1 = A.den;         // denominator of (A-1)

    // if topAminus1 == 0 => A=1
    if (topAminus1 == 0) {
        // Then (A-1) = 0 => eqn is 0*x0 + B=0 => B=0 => infinite solutions or no solutions
        // Usually B=0 => identity transform => not a meaningful Collatz step pattern
        if (B.num == 0) {
            // infinite solutions => not interesting
            return std::nullopt;
        } else {
            // no solutions
            return std::nullopt;
        }
    }

    // x_0 = -B / (A - 1)
    // Let B = bNum/bDen
    // Then x_0 = -(bNum/bDen) / ((topAminus1)/(botAminus1))
    //           = -(bNum/bDen) * (botAminus1 / topAminus1)
    //           = -(bNum * botAminus1) / (bDen * topAminus1)
    mpz_class bNum = B.num;
    mpz_class bDen = B.den;

    mpz_class x0num = -(bNum * botAminus1);
    mpz_class x0den = (bDen * topAminus1);

    // Now we must reduce x0num/x0den to see if it's an integer.
    // We'll get gcd, then divide.
    mpz_class g;
    mpz_gcd(g.get_mpz_t(), x0num.get_mpz_t(), x0den.get_mpz_t());
    x0num /= g;
    x0den /= g;

    // If denominator is ±1, we have an integer.
    if (x0den == 1) {
        return x0num;
    } else if (x0den == -1) {
        return -x0num;
    } else {
        // not an integer
        return std::nullopt;
    }
}

// Given a candidate integer x0, compute the forward sequence x0->x1->...->x_{k-1}
// according to 'pattern'. Ensure each x_i has the correct parity (odd/even).
// Finally, check x_k == x0.
bool verifyLoop(const mpz_class &x0, const std::vector<bool> &pattern) {
    // We’ll keep the values in mpz_class for correctness,
    // though steps can get large.
    std::vector<mpz_class> seq;
    seq.reserve(pattern.size()+1);
    seq.push_back(x0);

    for (bool isOdd : pattern) {
        mpz_class current = seq.back();
        bool actualOdd = (current % 2 != 0);
        if (actualOdd != isOdd) {
            return false; // parity mismatch
        }
        if (isOdd) {
            // next = 3*current + 1
            mpz_class nxt = 3*current + 1;
            seq.push_back(nxt);
        } else {
            // next = current / 2 (assuming current is even)
            // special check that current != 0
            if (current == 0) {
                // We'll bail out, as this leads nowhere interesting
                return false;
            }
            seq.push_back(current / 2);
        }
    }
    // Check x_k == x0
    return (seq.back() == x0);
}

int main() {
    // Increase kMax as you like, but be aware the search is O(2^k).
    // Also note that for large k, the numbers can explode quickly,
    // so running time & memory usage can become prohibitive.
    const int kMax = 16;

    bool foundAny = false;

    for (int k = 2; k <= kMax; ++k) {
        // 2^k parity patterns
        mpz_class patternsCount = (mpz_class(1) << k);
        for (mpz_class mask = 0; mask < patternsCount; mask++) {
            // Build pattern
            std::vector<bool> pattern(k);
            for (int i = 0; i < k; ++i) {
                bool bit = ((mask >> i) & 1) == 1;
                pattern[i] = bit;
            }
            // Compute linear form x_k = A*x_0 + B
            auto [A, B] = computeLinearForm(pattern);
            // Solve for x0
            auto maybeX0 = solveForX0(A, B);
            if (!maybeX0.has_value()) {
                continue;
            }
            mpz_class x0 = maybeX0.value();
            // Skip negative or zero x0 for standard Collatz loops
            if (x0 <= 0) {
                continue;
            }
            // Verify
            if (verifyLoop(x0, pattern)) {
                // Reconstruct the loop to print
                // but skip the trivial known loop [1->4->2->1] if it appears.
                // We'll gather the unique sorted elements (excluding the repeated last).
                // If they are exactly {1,2,4}, skip printing.
                std::vector<mpz_class> seq;
                seq.push_back(x0);
                for (bool isOdd : pattern) {
                    mpz_class cur = seq.back();
                    if (isOdd) {
                        seq.push_back(3*cur + 1);
                    } else {
                        seq.push_back(cur / 2);
                    }
                }
                // Now check if the sorted unique set is {1,2,4}
                std::vector<mpz_class> sortedUnique(seq.begin(), seq.end()-1); 
                std::sort(sortedUnique.begin(), sortedUnique.end(), [](auto &a, auto &b){return a < b;});
                sortedUnique.erase(std::unique(sortedUnique.begin(), sortedUnique.end()), sortedUnique.end());

                if (sortedUnique.size() == 3 &&
                    sortedUnique[0] == 1 && sortedUnique[1] == 2 && sortedUnique[2] == 4) {
                    continue; // skip known loop
                }

                // Print loop
                foundAny = true;
                std::cout << "Found loop of length " << k << ":\n";
                for (size_t i = 0; i < seq.size(); i++) {
                    std::cout << seq[i].get_str() 
                              << ((i+1 < seq.size()) ? " -> " : "\n");
                }
                // Print parity pattern
                std::cout << "Pattern: ";
                for (bool b : pattern) {
                    std::cout << (b ? "odd " : "even ");
                }
                std::cout << "\n\n";
            }
        }
    }

    if (!foundAny) {
        std::cout << "No new loops found for k up to " << kMax << ".\n";
    }
    return 0;
}
