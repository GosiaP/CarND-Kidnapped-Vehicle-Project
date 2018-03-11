/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;



void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // creates a normal (Gaussian) distribution for x,y, teta
  // define normal distributions for sensor noise
  normal_distribution<double> N_x(0, std[0]);
  normal_distribution<double> N_y(0, std[1]);
  normal_distribution<double> N_teta(0, std[2]);

  default_random_engine gen;

  // init particles
  for (int i = 0; i < num_particles; i++)
  {
    Particle p;
    p.id = i;
    p.x = x;
    p.y = y;
    p.theta = theta;
    p.weight = 1.0;

    // add noise
    p.x += N_x(gen);
    p.y += N_y(gen);
    p.theta += N_teta(gen);

    particles.push_back(p);
    std::cout << "particle {" << p.id << ", " << p.x << ", " << p.y << ", " << p.theta << "}";
  }

  is_initialized = true;
}



void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  // define normal distributions for sensor noise
  normal_distribution<double> N_x(0, std_pos[0]);
  normal_distribution<double> N_y(0, std_pos[1]);
  normal_distribution<double> N_theta(0, std_pos[2]);

  default_random_engine gen;

  for (std::vector<Particle>::iterator p = particles.begin(); p != particles.end(); ++p)
  {
    // calculate new state
    if (fabs(yaw_rate) < 0.00001) 
    {
      p->x += velocity * delta_t * cos(p->theta);
      p->y += velocity * delta_t * sin(p->theta);
    }
    else {
      p->x += velocity / yaw_rate * (sin(p->theta + yaw_rate * delta_t) - sin(p->theta));
      p->y += velocity / yaw_rate * (cos(p->theta) - cos(p->theta + yaw_rate * delta_t));
      p->theta += yaw_rate * delta_t;
    }

    // add noise
    p->x += N_x(gen);
    p->y += N_y(gen);
    p->theta += N_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) 
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (std::vector<LandmarkObs>::iterator o = observations.begin(); o != observations.end(); ++o) 
  {
    double min_dist = numeric_limits<double>::max();

    // init id of landmark from map placeholder to be associated with the observation
    int map_id = -1;
    for (std::vector<LandmarkObs>::const_iterator p = predicted.begin(); p != predicted.end(); ++p)
    {
      // get distance between current/predicted landmarks
      double cur_dist = dist(o->x, o->y, p->x, p->y);

      // find the predicted landmark nearest to the current observed landmark
      if (cur_dist < min_dist) 
      {
        min_dist = cur_dist;
        map_id = p->id;
      }
    }

    // set the observation's id to the nearest predicted landmark's id
    o->id = map_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  for (std::vector<Particle>::iterator p = particles.begin(); p != particles.end(); ++p)
  {
    // get the particle x, y coordinates
    double p_x = p->x;
    double p_y = p->y;
    double p_theta = p->theta;

    // create a vector to hold the map landmark locations predicted to be within sensor range of the particle
    std::vector<LandmarkObs> predictions;
    for (std::vector<Map::single_landmark_s>::const_iterator ml = map_landmarks.landmark_list.begin(); ml != map_landmarks.landmark_list.end(); ++ml)
    {

      // get id and x,y coordinates
      float lm_x = ml->x_f;
      float lm_y = ml->y_f;
      int   lm_id = ml->id_i;

      // only consider landmarks within sensor range of the particle (rather than using the "dist" method considering a circular 
      // region around the particle, this considers a rectangular region but is computationally faster)
      if ( fabs(lm_x - p_x) <= sensor_range
        && fabs(lm_y - p_y) <= sensor_range)
      {

        // add prediction to vector
        predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
      }
    }

    // create and populate a copy of the list of observations transformed from vehicle coordinates to map coordinates
    vector<LandmarkObs> transformed_os;
    for (std::vector<LandmarkObs>::const_iterator o = observations.begin(); o != observations.end(); ++o)
    {
      double t_x = cos(p_theta)*o->x - sin(p_theta)*o->y + p_x;
      double t_y = sin(p_theta)*o->x + cos(p_theta)*o->y + p_y;
      transformed_os.push_back(LandmarkObs{ o->id, t_x, t_y });
    }

    // perform dataAssociation for the predictions 
    // and transformed observations on this particle
    dataAssociation(predictions, transformed_os);

    // update weight for this particle:
    p->weight = 1.0;
    for (std::vector<LandmarkObs>::const_iterator tld = transformed_os.begin(); tld != transformed_os.end(); ++tld)
    {
      // placeholders for observation and associated prediction coordinates
      double o_x, o_y, pr_x, pr_y;
      o_x = tld->x;
      o_y = tld->y;

      int associated_prediction = tld->id;

      // get the x,y coordinates of the prediction associated with the current observation
      for (std::vector<LandmarkObs>::const_iterator ap = predictions.begin(); ap != predictions.end(); ++ap)
      {
        if (ap->id == associated_prediction)
        {
          pr_x = ap->x;
          pr_y = ap->y;
        }
      }

      // calculate weight for this observation with multivariate Gaussian
      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double obs_w = (1 / (2 * M_PI*s_x*s_y)) * exp(-(pow(pr_x - o_x, 2) / (2 * pow(s_x, 2)) + (pow(pr_y - o_y, 2) / (2 * pow(s_y, 2)))));

      // total observations weight
      p->weight *= obs_w;
    }
  }
}

void ParticleFilter::resample()
{
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // find max weight
  vector<double> weights;
  for (int i = 0; i < num_particles; i++)
  {
    weights.push_back(particles[i].weight);
  }
  const double max_weight = *max_element(weights.begin(), weights.end());

  // generate random starting index for resampling wheel
  uniform_int_distribution<int> uni_int_dist(0, num_particles - 1);
  // uniform random distribution [0.0, max_weight)
  uniform_real_distribution<double> uni_real_dist(0.0, max_weight);

  default_random_engine gen;
  auto index = uni_int_dist(gen);

  // spin the resample wheel
  double beta = 0.0;
  vector<Particle> new_particles;
  for (int i = 0; i < num_particles; i++) 
  {
    beta += uni_real_dist(gen) * 2.0;
    while (beta > weights[index]) 
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
