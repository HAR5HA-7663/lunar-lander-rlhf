package com.lunarlander.tracker.controller;

import com.lunarlander.tracker.model.Experiment;
import com.lunarlander.tracker.repository.ExperimentRepository;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.*;

/**
 * REST endpoints for querying RLHF experiment results.
 *
 * All endpoints are read-only — Python notebooks write the data.
 */
@RestController
@RequestMapping("/api/experiments")
@CrossOrigin(origins = "*")
public class ExperimentController {

    private final ExperimentRepository repo;

    public ExperimentController(ExperimentRepository repo) {
        this.repo = repo;
    }

    /** GET /api/experiments — all experiments, newest first. */
    @GetMapping
    public List<Experiment> getAll() {
        return repo.findAllByOrderByTimestampDesc();
    }

    /** GET /api/experiments/{id} — single experiment by id. */
    @GetMapping("/{id}")
    public ResponseEntity<Experiment> getById(@PathVariable Long id) {
        return repo.findById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    /** GET /api/experiments/type/{expType} — filter by type. */
    @GetMapping("/type/{expType}")
    public List<Experiment> getByType(@PathVariable String expType) {
        return repo.findByExpTypeOrderByTimestampDesc(expType);
    }

    /** GET /api/experiments/latest — most recent experiment. */
    @GetMapping("/latest")
    public ResponseEntity<Experiment> getLatest() {
        return repo.findTopByOrderByTimestampDesc()
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    /**
     * GET /api/experiments/compare — grouped summary comparing baseline vs aligned.
     * Returns a map with 'baseline_ppo' and 'aligned_ppo' keys containing summary stats.
     */
    @GetMapping("/compare")
    public Map<String, Object> compare() {
        List<Experiment> trainingExps = repo.findLatestTrainingExperiments();

        Map<String, Object> result = new LinkedHashMap<>();
        Map<String, Map<String, Object>> grouped = new LinkedHashMap<>();

        for (Experiment e : trainingExps) {
            // Keep only the latest per type
            grouped.computeIfAbsent(e.getExpType(), k -> {
                Map<String, Object> summary = new LinkedHashMap<>();
                summary.put("name", e.getName());
                summary.put("timestamp", e.getTimestamp());
                summary.put("mean_reward", e.getMeanReward());
                summary.put("std_reward", e.getStdReward());
                summary.put("success_rate", e.getSuccessRate());
                summary.put("crash_rate", e.getCrashRate());
                summary.put("mean_ep_len", e.getMeanEpLen());
                return summary;
            });
        }

        result.put("experiments", grouped);
        result.put("total_logged", repo.count());

        // Compute delta if both types are present
        if (grouped.containsKey("baseline_ppo") && grouped.containsKey("aligned_ppo")) {
            Object bRaw = grouped.get("baseline_ppo").get("mean_reward");
            Object aRaw = grouped.get("aligned_ppo").get("mean_reward");
            if (bRaw instanceof Number b && aRaw instanceof Number a) {
                Map<String, Object> delta = new LinkedHashMap<>();
                delta.put("mean_reward_delta", a.doubleValue() - b.doubleValue());
                double bSR = bRaw instanceof Number ? ((Number) grouped.get("baseline_ppo").get("success_rate")).doubleValue() : 0;
                double aSR = aRaw instanceof Number ? ((Number) grouped.get("aligned_ppo").get("success_rate")).doubleValue() : 0;
                delta.put("success_rate_delta", aSR - bSR);
                result.put("delta_aligned_vs_baseline", delta);
            }
        }

        return result;
    }
}
