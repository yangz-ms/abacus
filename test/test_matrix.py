import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from calc import calc11
from test_helper import test

if __name__ == '__main__':
    # calc11: Matrix operations
    test("det([[1,2],[3,4]])", "-2", calc11)
    test("det([[1,0,0],[0,1,0],[0,0,1]])", "1", calc11)
    test("trace([[1,2],[3,4]])", "5", calc11)
    test("dot([1,2,3],[4,5,6])", "32", calc11)
    test("cross([1,0,0],[0,1,0])", "[0,0,1]", calc11)

    # calc11: Matrix arithmetic
    test("[[1,2],[3,4]]+[[5,6],[7,8]]", "[[6,8],[10,12]]", calc11)
    test("[[1,2],[3,4]]*[[1,0],[0,1]]", "[[1,2],[3,4]]", calc11)
    test("[[1,2],[3,4]]*[[5,6],[7,8]]", "[[19,22],[43,50]]", calc11)
    test("[[5,6],[7,8]]-[[1,2],[3,4]]", "[[4,4],[4,4]]", calc11)

    # calc11: Inverse
    test("inv([[2,1],[1,1]])", "[[1,-1],[-1,2]]", calc11)

    # calc11: Scalar operations
    test("2*[[1,2],[3,4]]", "[[2,4],[6,8]]", calc11)
    test("[[2,4],[6,8]]/2", "[[1,2],[3,4]]", calc11)

    # calc11: Transpose and RREF
    test("trans([[1,2,3],[4,5,6]])", "[[1,4],[2,5],[3,6]]", calc11)
    test("rref([[1,2,3],[4,5,6]])", "[[1,0,-1],[0,1,2]]", calc11)

    # calc11: Matrix power
    test("[[1,1],[0,1]]^3", "[[1,3],[0,1]]", calc11)
