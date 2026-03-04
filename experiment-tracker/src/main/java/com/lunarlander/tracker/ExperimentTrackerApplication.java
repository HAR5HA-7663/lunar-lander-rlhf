package com.lunarlander.tracker;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;

/**
 * LunarLander RLHF Experiment Tracker
 *
 * Spring Boot REST API that reads experiment results from the SQLite database
 * shared with Python training notebooks.
 *
 * Start: ./mvnw spring-boot:run  (requires Java 17+)
 * API:   http://localhost:8080/api/experiments
 */
@SpringBootApplication
@EnableJpaRepositories(basePackages = "com.lunarlander.tracker.repository")
public class ExperimentTrackerApplication {

    public static void main(String[] args) {
        SpringApplication.run(ExperimentTrackerApplication.class, args);
    }
}
