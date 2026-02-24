from calc import calc15
from test_helper import test

if __name__ == '__main__':
    # Linear inequalities
    test("2*x+3>7", "(2,inf)", calc15)
    test("3-x>=1", "(-inf,2]", calc15)
    test("x<5", "(-inf,5)", calc15)
    test("2*x>=4", "[2,inf)", calc15)
    test("x+1>0", "(-1,inf)", calc15)
    test("5*x<=10", "(-inf,2]", calc15)

    # Quadratic inequalities
    test("x^2-4>0", "(-inf,-2) U (2,inf)", calc15)
    test("x^2-4<0", "(-2,2)", calc15)
    test("x^2-4<=0", "[-2,2]", calc15)
    test("x^2-4>=0", "(-inf,-2] U [2,inf)", calc15)
    test("x^2+1<0", "no solution", calc15)
    test("x^2+1>0", "(-inf,inf)", calc15)
    test("x^2-1<=0", "[-1,1]", calc15)

    # Cubic inequalities
    test("x^3-x>0", "(-1,0) U (1,inf)", calc15)
    test("x^3-x<0", "(-inf,-1) U (0,1)", calc15)

    # Absolute value inequalities
    test("abs(x-3)<5", "(-2,8)", calc15)
    test("abs(x)>=2", "(-inf,-2] U [2,inf)", calc15)
    test("abs(x-2)<=5", "[-3,7]", calc15)
    test("abs(x)<3", "(-3,3)", calc15)
    test("abs(x)>0", "(-inf,0) U (0,inf)", calc15)

    # Compound inequalities
    test("1<2*x+3<7", "(-1,2)", calc15)
    test("0<=x<=5", "[0,5]", calc15)

    # Constant inequalities
    test("3>2", "(-inf,inf)", calc15)
    test("1>2", "no solution", calc15)

    # Conic sections - circle
    test("conic(x^2+y^2-25)", "Circle: x^2+y^2=25, center=(0,0), radius=5", calc15)
    test("conic(x^2+y^2-6*x+4*y-12)", "Circle: (x-3)^2+(y+2)^2=25, center=(3,-2), radius=5", calc15)

    # Conic sections - ellipse
    test("conic(x^2/9+y^2/4-1)", "Ellipse: x^2/9+y^2/4=1, center=(0,0), a=3, b=2", calc15)
    test("conic(x^2+4*y^2-16)", "Ellipse: x^2/16+y^2/4=1, center=(0,0), a=4, b=2", calc15)

    # Conic sections - hyperbola
    test("conic(x^2-y^2-1)", "Hyperbola: x^2/1-y^2/1=1, center=(0,0), a=1, b=1", calc15)
    test("conic(x^2/16-y^2/9-1)", "Hyperbola: x^2/16-y^2/9=1, center=(0,0), a=4, b=3", calc15)

    # Conic sections - parabola
    test("conic(y^2-4*x)", "Parabola: y^2=4*(x-0), vertex=(0,0), p=1, opens right", calc15)
    test("conic(x^2+2*y)", "Parabola: x^2=-2*(y-0), vertex=(0,0), p=-0.5, opens down", calc15)

    # Fallthrough to parent calculator
    test("2+3", "5", calc15)
    test("x+1", "x+1", calc15)
    test("2*x+3*x", "5*x", calc15)
