package com.lunarlander.tracker.model;

import jakarta.persistence.*;

/**
 * JPA entity mapping the 'experiments' table created by Python's db_logger.py.
 *
 * Schema:
 *   id, name, exp_type, timestamp, mean_reward, std_reward,
 *   success_rate, crash_rate, mean_ep_len, hyperparams (JSON), notes
 */
@Entity
@Table(name = "experiments")
public class Experiment {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String name;

    @Column(name = "exp_type", nullable = false)
    private String expType;

    @Column(nullable = false)
    private String timestamp;

    @Column(name = "mean_reward")
    private Double meanReward;

    @Column(name = "std_reward")
    private Double stdReward;

    @Column(name = "success_rate")
    private Double successRate;

    @Column(name = "crash_rate")
    private Double crashRate;

    @Column(name = "mean_ep_len")
    private Double meanEpLen;

    @Column(columnDefinition = "TEXT")
    private String hyperparams;

    @Column(columnDefinition = "TEXT")
    private String notes;

    // --- Getters ---

    public Long getId() { return id; }
    public String getName() { return name; }
    public String getExpType() { return expType; }
    public String getTimestamp() { return timestamp; }
    public Double getMeanReward() { return meanReward; }
    public Double getStdReward() { return stdReward; }
    public Double getSuccessRate() { return successRate; }
    public Double getCrashRate() { return crashRate; }
    public Double getMeanEpLen() { return meanEpLen; }
    public String getHyperparams() { return hyperparams; }
    public String getNotes() { return notes; }
}
