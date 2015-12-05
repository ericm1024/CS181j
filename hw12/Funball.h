// -*- C++ -*-

#ifndef FUNBALL_H
#define FUNBALL_H

#include <vector>

namespace Funball {

// it's like claremont cash, but worthless
struct MuddMoney {
  unsigned int _idWithinTheDorm;
  unsigned int _dormNumber;
  double _amountOfMoney;
};

std::vector<MuddMoney>
initializeFun(const unsigned int numberOfStudentsPerDorm,
              const unsigned long long randomNumberSeed);

void
checkFunballAnswer(const unsigned int numberOfStudentsPerDorm,
                   const std::vector<MuddMoney> & thisRanksStartingMuddMoneys,
                   const std::vector<MuddMoney> & thisRanksFinalMuddMoneys);

}

#endif // FUNBALL_H
