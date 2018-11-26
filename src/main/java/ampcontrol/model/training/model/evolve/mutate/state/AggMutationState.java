package ampcontrol.model.training.model.evolve.mutate.state;

import java.util.Objects;

/**
 * {@link MutationState} which aggregates several {@link MutationState}s.
 * @param <T>
 *
 * @author Christian Skärby
 */
public class AggMutationState<T,S> implements MutationState<T,S> {

    private final MutationState<T,S> first;
    private final MutationState<T,S> second;

    /**
     * Creates a new {@link Builder} instance
     *
     * @return a new instance
     */
    public static <T,S> Builder<T,S> builder() {
        return new Builder<>();
    }

    /**
     * Constructor
     *
     * @param first  First MutationState to apply
     * @param second Second MutationState to apply
     */
    public AggMutationState(MutationState<T,S> first, MutationState<T,S> second) {
        this.first = first;
        this.second = second;
    }

    @Override
    public T mutate(T toMutate, S state) {
        return second.mutate(first.mutate(toMutate, state), state);
    }

    public static class Builder<T, S> {
        private MutationState<T, S> first;
        private MutationState<T, S> second;

        /**
         * Sets the first MutationState
         *
         * @param first a first MutationState
         * @return the builder
         */
        public Builder<T,S> first(MutationState<T,S> first) {
            this.first = first;
            return this;
        }

        /**
         * Sets the second MutationState
         *
         * @param second a second MutationState
         * @return the builder
         */
        public Builder<T,S> second(MutationState<T,S> second) {
            this.second = second;
            return this;
        }

        /**
         * Convenience method for adding more MutationStates. Will create new {@link AggMutationState}s
         * and {@link Builder}s when all MutationStates are set.
         *
         * @param next a next MutationState
         * @return a builder (might not be the same as call was made to)
         */
        public Builder<T,S> andThen(MutationState<T,S> next) {
            if (first == null) {
                return first(next);
            }
            if (second == null) {
                return second(next);
            }
            return AggMutationState.<T,S>builder()
                    .first(build())
                    .second(next);
        }

        /**
         * Builds a new {@link AggMutationState}
         *
         * @return a new {@link AggMutationState}
         */
        public AggMutationState<T,S> build() {
            return new AggMutationState<>(Objects.requireNonNull(first), Objects.requireNonNull(second));
        }
    }
}
