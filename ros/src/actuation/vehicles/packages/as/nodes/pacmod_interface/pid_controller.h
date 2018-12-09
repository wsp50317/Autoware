#ifndef PID_CONTROLLER_H
#define PID_CONTROLLER_H

class PIDController
{
public:
  PIDController()
  : prev_e_(0.0)
  , i_(0.0)
  , kp_(0.0)
  , ki_(0.0)
  , kd_(0.0)
  , min_(0.0)
  , max_(0.0)
  {
  }

  void setGain(double kp, double ki, double kd)
  {
    kp_ = kp;
    ki_ = ki;
    kd_ = kd;
  }

  void setMinMax(double min, double max)
  {
    min_ = min;
    max_ = max;
  }

  void reset()
  {
    i_ = 0.0;
  }

  double update(double e, double dt)
  {
    static double p, i, d;

    p = e;
    i = i_ + e * dt;
    d = (e - prev_e_) / dt;

    double x = kp_ * p + ki_ * i_ + kd_ * d;

    if (min_ < x && x < max_)
    {
      i_ = i;
    }

    x = std::max(x, min_);
    x = std::min(x, max_);

    prev_e_ = e;

    return x;
  }

private:
  double prev_e_, i_;
  double kp_, ki_, kd_;
  double min_, max_;
};

#endif  // PID_CONTROLLER_H
