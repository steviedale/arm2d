#include "arm2d/polygon.h"
#include <deque>

Polygon::Polygon(std::vector<Point> points)
{
  BoostPolygon* p = this;
  boost::geometry::assign_points(*p, points);
}

bool Polygon::intersects(Polygon &other)
{
  BoostPolygon* this_polygon = this;
  BoostPolygon* other_polygon = &other;
  std::deque<BoostPolygon> output;
  return boost::geometry::intersection(*this_polygon, *other_polygon, output); 
}