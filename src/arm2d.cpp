#include "arm2d/polygon.h"
#include <iostream>


int main(int argc, char **argv)
{
  std::vector<Point> points = {Point(1,0), Point(2,2), Point(3,3)};
  Polygon p(points);
  std::cout << p.intersects(p) << std::endl; 
  std::cout << "it worked!" << std::endl; 
}