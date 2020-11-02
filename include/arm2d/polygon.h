#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

typedef boost::geometry::model::d2::point_xy<double> Point;
typedef boost::geometry::model::polygon<Point> BoostPolygon;

class Polygon : public BoostPolygon
{
public:
  Polygon(std::vector<Point> points);
  bool intersects(Polygon &other);
};