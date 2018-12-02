package ampcontrol.model.training.model.evolve.fitness;

import org.slf4j.Logger;

import java.util.function.Consumer;
import java.util.function.DoubleUnaryOperator;

/**
 * A policy for measuring fitness of candidates.
 *
 * @param <T>
 * @author Christian Skärby
 */
public interface FitnessPolicy<T> {

    /**
     * Apply the scoring policy to the given candidate
     * @param candidate Candidate for which fitness score shall be measured
     * @param fitnessListener Listens to fitness score
     * @return The candidate for which the fitness policy has been applied
     */
    T apply(T candidate, Consumer<Double> fitnessListener);


    /**
     * Returns a builder for decorating the given policy
     * @param policy {@link FitnessPolicy} to decorate
     * @param <T> Type of policy
     * @return a new {@link DecorationBuilder} instance
     */
    static <T> DecorationBuilder<T> decorate(FitnessPolicy<T> policy) {
        return new DecorationBuilder<>(policy);
    }

    class DecorationBuilder<T> {

        private FitnessPolicy<T> policy;

        private DecorationBuilder(FitnessPolicy<T> policy) {
            this.policy = policy;
        }

        /**
         * Adds a transformation for fitness values, e.g. a scaling
         * @param transform transforms fitness
         * @return the Builder for fluent API
         */
        public DecorationBuilder<T> transform(DoubleUnaryOperator transform) {
            this.policy = new TransformFitness<>(transform, policy);
            return this;
        }

        /**
         * Enables logging for the last added policy. Note that certain method calls will wrap the policy in other
         * policies. Make sure this is called where it is desirable
         * @return the Builder for fluent API
         */
        public DecorationBuilder<T> log() {
            this.policy = new LogFitness<>(policy);
            return this;
        }

        /**
         * Enables logging for the last added policy. Note that certain method calls will wrap the policy in other
         * policies. Make sure this is called where it is desirable
         * @param logger logger to use
         * @return the Builder for fluent API
         */
        public DecorationBuilder<T> log(Logger logger) {
            this.policy = new LogFitness<>(logger, policy);
            return this;
        }

        /**
         * Accumlates a given number of samples before reporting the average as the fitness
         * @param nrofSamplesToAverage number of samples to average
         * @return the Builder for fluent API
         */
        public DecorationBuilder<T> average(int nrofSamplesToAverage) {
            policy = new TransformFitness<>(
                    d -> d / nrofSamplesToAverage,
                    new AccumulateFitness<>(nrofSamplesToAverage,
                            (d1,d2) -> d1+d2,
                            0,
                            policy));
            return this;
        }

        /**
         * Decoration complete, return the new {@link FitnessPolicy}
         * @return the new {@link FitnessPolicy}
         */
        public FitnessPolicy<T> done() {
            return policy;
        }

    }

}
