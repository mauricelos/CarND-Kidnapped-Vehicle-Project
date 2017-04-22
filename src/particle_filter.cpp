/*
 * particle_filter.cpp
 *
 *  Created on: April 18, 2017
 *      Author: Maurice Loskyll
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    
    //initialization of the particle filter
    num_particles = 50;
    weights.resize(num_particles);
    default_random_engine generate;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    
    for (int i = 0; i < num_particles; ++i) {
        
        Particle particle;
        particle.x = dist_x(generate);
        particle.y = dist_y(generate);
        particle.theta = dist_theta(generate);
        particle.weight = 1;
        particles.push_back(particle);
    }
    
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    
    //implementation of the bicycle model plus noise
    default_random_engine generate;
    normal_distribution<double> noise_x(0, std_pos[0]);
    normal_distribution<double> noise_y(0, std_pos[1]);
    normal_distribution<double> noise_theta(0, std_pos[2]);
    
    //checking if yaw_rate is zero and setting to new value if so
    if (fabs(yaw_rate) < 0.0001) {
        
        yaw_rate = 0.0001;
    }
    
    for (auto&& particle : particles){
        
        particle.x += (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta)) + noise_x(generate);
        particle.y += (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t)) + noise_y(generate);
        particle.theta += yaw_rate * delta_t + noise_theta(generate);
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], vector<LandmarkObs> observations, Map map_landmarks) {
    
    //transformation of the observations to map space and calculating of the weight multiplier
    for (int i = 0; i < num_particles; i++) {
        
        long double weight = 1;
        double transformed_obs_x = 0;
        double transformed_obs_y = 0;
        
        //transforming the observations to map space
        for (int j = 0; j < observations.size(); j++) {
            
            double cos_theta = cos(particles[i].theta);
            double sin_theta = sin(particles[i].theta);
            
            transformed_obs_x = observations[j].x * cos_theta - observations[j].y * sin_theta + particles[i].x;
            transformed_obs_y = observations[j].y * cos_theta + observations[j].x * sin_theta + particles[i].y;
            
            Map::single_landmark_s nearest_landmark;
            double min_distance = sensor_range;
            double distance = 0;
            
            //calculating the distance between landmarks and transformed observations
            for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
                
                Map::single_landmark_s landmark = map_landmarks.landmark_list[k];
                
                distance = fabs(transformed_obs_x - landmark.x_f) + fabs(transformed_obs_y - landmark.y_f);
                
                //finding the nearest landmark for the transformed observations
                if (distance < min_distance) {
                    min_distance = distance;
                    nearest_landmark = landmark;
                }
            }
            
            //calcualting the bivariate gaussian distribution for the weight_multiplier
            double ox = transformed_obs_x;
            double oy = transformed_obs_y;
            double lx = nearest_landmark.x_f;
            double ly = nearest_landmark.y_f;
            
            long double nominator = exp(-0.5 * (((lx - ox) * (lx - ox)) / (std_landmark[0] * std_landmark[0]) + ((ly - oy) * (ly - oy)) / (std_landmark[1] * std_landmark[1])));
            long double denominator = 2 * M_PI * std_landmark[0] * std_landmark[1];
            long double weight_multiplier = nominator / denominator;
            
            weight *= weight_multiplier;
        }
        
        particles[i].weight = weight;
        weights[i] = weight;
    }
}

void ParticleFilter::resample() {
    
    //resample particles with discrete distribution
    vector<Particle> resampled_particles;
    
    default_random_engine generate;
    discrete_distribution<> distribution(weights.begin(), weights.end());
    
    for (int i = 0; i < num_particles; ++i) {
        
        int number = distribution(generate);
        resampled_particles.push_back(particles[number]);
    }
    
    particles = resampled_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
