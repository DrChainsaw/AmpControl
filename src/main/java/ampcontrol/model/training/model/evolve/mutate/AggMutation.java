package ampcontrol.model.training.model.evolve.mutate;

import java.util.Objects;

/**
 * Aggregates two {@link Mutation}s, applying both to any input
 *
 * @author Christian Skärby
 */
public class AggMutation<T> implements Mutation<T> {

    private final Mutation<T> first;
    private final Mutation<T> second;

    /**
     * Creates a new {@link Builder} instance
     *
     * @return a new instance
     */
    public static <T> Builder<T> builder() {
        return new Builder<>();
    }

    /**
     * Constructor
     *
     * @param first  First mutation to apply
     * @param second Second mutation to apply
     */
    public AggMutation(Mutation<T> first, Mutation<T> second) {
        this.first = first;
        this.second = second;
    }

    @Override
    public T mutate(T toMutate) {
        return second.mutate(first.mutate(toMutate));
    }

    public static class Builder<T> {
        private Mutation<T> first;
        private Mutation<T> second;

        /**
         * Sets the first mutation
         *
         * @param first a first mutation
         * @return the builder
         */
        public Builder<T> first(Mutation<T> first) {
            this.first = first;
            return this;
        }

        /**
         * Sets the second mutation
         *
         * @param second a second mutation
         * @return the builder
         */
        public Builder<T> second(Mutation<T> second) {
            this.second = second;
            return this;
        }

        /**
         * Convenience method for adding more mutations. Will create new {@link AggMutation}s
         * and {@link Builder}s when all Mutations are set.
         *
         * @param next a next mutation
         * @return a builder (might not be the same as call was made to)
         */
        public Builder<T> andThen(Mutation<T> next) {
            if (first == null) {
                return first(next);
            }
            if (second == null) {
                return second(next);
            }
            return AggMutation.<T>builder()
                    .first(build())
                    .second(next);
        }

        /**
         * Builds a new {@link AggMutation}
         *
         * @return a new {@link AggMutation}
         */
        public AggMutation<T> build() {
            return new AggMutation<>(Objects.requireNonNull(first), Objects.requireNonNull(second));
        }
    }
}
