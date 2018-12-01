package ampcontrol.model.training.model.evolve.fitness;

import java.util.LinkedHashSet;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * Combines fitness from two {@link FitnessPolicy FitnessPolicies} using a provided combiner
 *
 * @param <T>
 * @author Christian Skärby
 */
public class CombinePolicy<T> implements FitnessPolicy<T> {

    private final DoubleBinaryOperator combiner;
    private final Set<FitnessState<T>> monitoredFitness;
    private final Set<FitnessState<T>> reportedFitness = new LinkedHashSet<>();
    private Consumer<Double> actualListener;

    private final static class FitnessState<T> implements Consumer<Double> {

        private final DoubleBinaryOperator aggregationMethod;
        private final DoubleUnaryOperator transform;
        private final FitnessPolicy<T> policy;
        private final Set<Double> fitness = new LinkedHashSet<>();
        private Consumer<FitnessState<T>> fitnessCallback;

        private FitnessState(DoubleBinaryOperator aggregationMethod, DoubleUnaryOperator transform, FitnessPolicy<T> policy) {
            this.aggregationMethod = aggregationMethod;
            this.transform = transform;
            this.policy = policy;
        }

        @Override
        public void accept(Double fitness) {
            this.fitness.add(transform.applyAsDouble(fitness));
            fitnessCallback.accept(this);
        }

        private double getFitness() {
            return fitness.stream().mapToDouble(d->d).reduce(aggregationMethod).orElse(Double.MAX_VALUE);
        }
    }

    /**
     * Returns a builder for the policy
     * @param <T> type of FitnessPolicy
     * @return a new Builder instance
     */
    public static <T> Builder<T> builder() {
        return new Builder<>();
    }

    /**
     * Constructor. Use Builder to create
     * @param combiner Method for combining fitness values
     * @param monitoredFitness Fitness to combine
     */
    private CombinePolicy(DoubleBinaryOperator combiner, Set<FitnessState<T>> monitoredFitness) {
        this.monitoredFitness = monitoredFitness;
        this.combiner = combiner;
        monitoredFitness.forEach(fs -> fs.fitnessCallback = this::fitessCallback);
    }


    @Override
    public T apply(T candidate, Consumer<Double> fitnessListener) {
        actualListener = fitnessListener;
        return monitoredFitness.stream().reduce(
                candidate,
                (cand, fitnessState) -> fitnessState.policy.apply(cand, fitnessState),
                (c1, c2) -> c2);
    }

    private void fitessCallback(FitnessState<T> state) {
        if (!monitoredFitness.contains(state)) {
            throw new IllegalArgumentException("Got unknown FitnessState: " + state + " monitored: " + monitoredFitness);
        }
        reportedFitness.add(state);
        if (reportedFitness.size() == monitoredFitness.size()) {
            final double fitness = reportedFitness.stream()
                    .mapToDouble(FitnessState::getFitness)
                    .reduce(combiner).orElse(Double.MAX_VALUE);

            reportedFitness.forEach(fs -> fs.fitness.clear());
            reportedFitness.clear();
            actualListener.accept(fitness);
        }
    }

    public static final class Builder<T> {
        private final Set<FitnessState<T>> monitoredFitness = new LinkedHashSet<>();
        private DoubleBinaryOperator combiner = (d1,d2) -> d1+d2;

        public static final class StateBuilder<T> {
            private final Builder<T> builder;
            private final FitnessPolicy<T> policy;
            private DoubleBinaryOperator aggregationMethod = (d1,d2) -> d1+d2;
            private DoubleUnaryOperator transform = DoubleUnaryOperator.identity();

            private StateBuilder(Builder<T> builder, FitnessPolicy<T> policy) {
                this.builder = builder;
                this.policy = policy;
            }

            /**
             * Determines how multiple fitness values from the same policy shall be combined
             * @param aggregationMethod Aggregates multiple fitness values
             * @return the Builder for fluent API
             */
            public StateBuilder<T> aggregationMethod(DoubleBinaryOperator aggregationMethod) {
                this.aggregationMethod = aggregationMethod;
                return this;
            }

            /**
             * Adds a transformation for fitness values, e.g. a scaling
             * @param transform transforms fitness
             * @return the Builder for fluent API
             */
            public StateBuilder<T> transform(DoubleUnaryOperator transform) {
                this.transform = transform;
                return this;
            }

            /**
             * Add {@link FitnessPolicy}
             * @return a new StateBuilder
             */
            public StateBuilder<T> add(FitnessPolicy<T> policy) {
                builder.monitoredFitness.add(new FitnessState<>(aggregationMethod, transform, this.policy));
                return new StateBuilder<>(builder, policy);
            }

            /**
             * Builds a new {@link CombinePolicy} instance
             * @return a new instance
             */
            public CombinePolicy<T> build() {
                builder.monitoredFitness.add(new FitnessState<>(aggregationMethod, transform, policy));
                return builder.build();
            }
        }

        private Builder() {
            /* Constructor */
        }

        /**
         * Determines how fitness from different policies shall be combined
         * @param combiner Combines fitness
         * @return the Builder for fluent API
         */
        public Builder<T> combiner(DoubleBinaryOperator combiner) {
            this.combiner = combiner;
            return this;
        }

        /**
         * Add {@link FitnessPolicy}
         * @return a new StateBuilder
         */
        public StateBuilder<T> add(FitnessPolicy<T> policy) {
            return new StateBuilder<>(this, policy);
        }

        /**
         * Builds a new {@link CombinePolicy} instance
         * @return a new instance
         */
        public CombinePolicy<T> build() {
            return new CombinePolicy<>(combiner, new LinkedHashSet<>(monitoredFitness));
        }
    }
}
