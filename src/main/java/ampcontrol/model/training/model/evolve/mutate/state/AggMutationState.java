package ampcontrol.model.training.model.evolve.mutate.state;

import java.io.IOException;
import java.util.Objects;

/**
 * {@link MutationState} which aggregates several {@link MutationState}s.
 * @param <T>
 *
 * @author Christian Skärby
 */
public class AggMutationState<T> implements MutationState<T> {

    private final MutationState<T> first;
    private final MutationState<T> second;

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
     * @param first  First MutationState to apply
     * @param second Second MutationState to apply
     */
    public AggMutationState(MutationState<T> first, MutationState<T> second) {
        this.first = first;
        this.second = second;
    }

    @Override
    public T mutate(T toMutate) {
        return second.mutate(first.mutate(toMutate));
    }

    @Override
    public void save(String baseName) throws IOException {
        first.save(baseName);
        second.save(baseName);
    }

    @Override
    public MutationState<T> clone() {
        return AggMutationState.<T>builder().first(first.clone()).second(second.clone()).build();
    }

    public static class Builder<T> {
        private MutationState<T> first;
        private MutationState<T> second;

        /**
         * Sets the first MutationState
         *
         * @param first a first MutationState
         * @return the builder
         */
        public Builder<T> first(MutationState<T> first) {
            this.first = first;
            return this;
        }

        /**
         * Sets the second MutationState
         *
         * @param second a second MutationState
         * @return the builder
         */
        public Builder<T> second(MutationState<T> second) {
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
        public Builder<T> andThen(MutationState<T> next) {
            if (first == null) {
                return first(next);
            }
            if (second == null) {
                return second(next);
            }
            return AggMutationState.<T>builder()
                    .first(build())
                    .second(next);
        }

        /**
         * Builds a new {@link AggMutationState}
         *
         * @return a new {@link AggMutationState}
         */
        public AggMutationState<T> build() {
            return new AggMutationState<>(Objects.requireNonNull(first), Objects.requireNonNull(second));
        }
    }
}
