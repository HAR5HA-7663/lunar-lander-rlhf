package com.lunarlander.tracker.repository;

import com.lunarlander.tracker.model.Experiment;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface ExperimentRepository extends JpaRepository<Experiment, Long> {

    /** Filter experiments by type (baseline_ppo, reward_model, aligned_ppo). */
    List<Experiment> findByExpTypeOrderByTimestampDesc(String expType);

    /** Most recently logged experiment. */
    Optional<Experiment> findTopByOrderByTimestampDesc();

    /** All experiments ordered newest first. */
    List<Experiment> findAllByOrderByTimestampDesc();

    /** Comparison: latest of each training type that has a mean_reward. */
    @Query(value = """
        SELECT * FROM experiments
        WHERE exp_type IN ('baseline_ppo', 'aligned_ppo')
          AND mean_reward IS NOT NULL
        ORDER BY exp_type, timestamp DESC
        """, nativeQuery = true)
    List<Experiment> findLatestTrainingExperiments();
}
