#include "../include/stylor.hpp"
#include <gtest/gtest.h>

/// \brief Verify the test framework is functional.
TEST(StylorBase, AssertTrue) { ASSERT_TRUE(true); }

/// \brief Entry point for Google Test.
/// \param argc Argument count.
/// \param argv Argument vector.
/// \return Google Test execution result.
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
