package ampcontrol.model.training.model.evolve.fitness;


import java.util.Objects;
import java.util.function.Consumer;

/**
 * Aggregates two {@link FitnessPolicy FitnessPolicies}, applying both to any input
 *
 * @author Christian Skärby
 */
public class AggPolicy<T> implements FitnessPolicy<T> {

    private final FitnessPolicy<T> first;
    private final FitnessPolicy<T> second;

    /**
     * Creates a new {@link AggPolicy.Builder} instance
     *
     * @return a new instance
     */
    public static <T> AggPolicy.Builder<T> builder() {
        return new AggPolicy.Builder<>();
    }

    /**
     * Constructor
     *
     * @param first  First FitnessPolicy to apply
     * @param second Second FitnessPolicy to apply
     */
    private AggPolicy(FitnessPolicy<T> first, FitnessPolicy<T> second) {
        this.first = first;
        this.second = second;
    }

    @Override
    public T apply(T candidate, Consumer<Double> fitnessListener) {
        return second.apply(first.apply(candidate, fitnessListener), fitnessListener);
    }


    public static class Builder<T> {
        private FitnessPolicy<T> first;
        private FitnessPolicy<T> second;

        /**
         * Sets the first FitnessPolicy
         *
         * @param first a first FitnessPolicy
         * @return the builder
         */
        public AggPolicy.Builder<T> first(FitnessPolicy<T> first) {
            this.first = first;
            return this;
        }

        /**
         * Sets the second FitnessPolicy
         *
         * @param second a second FitnessPolicy
         * @return the builder
         */
        public AggPolicy.Builder<T> second(FitnessPolicy<T> second) {
            this.second = second;
            return this;
        }

        /**
         * Convenience method for adding more FitnessPolicys. Will create new {@link AggPolicy}s
         * and {@link AggPolicy.Builder}s when all FitnessPolicys are set.
         *
         * @param next a next FitnessPolicy
         * @return a builder (might not be the same as call was made to)
         */
        public AggPolicy.Builder<T> andThen(FitnessPolicy<T> next) {
            if (first == null) {
                return first(next);
            }
            if (second == null) {
                return second(next);
            }
            return AggPolicy.<T>builder()
                    .first(build())
                    .second(next);
        }

        /**
         * Builds a new {@link AggPolicy}
         *
         * @return a new {@link AggPolicy}
         */
        public AggPolicy<T> build() {
            return new AggPolicy<>(Objects.requireNonNull(first), Objects.requireNonNull(second));
        }
    }
}
