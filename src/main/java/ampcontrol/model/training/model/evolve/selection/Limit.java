package ampcontrol.model.training.model.evolve.selection;

import java.util.List;
import java.util.Map;
import java.util.function.LongSupplier;
import java.util.stream.Stream;

/**
 * {@link Selection} which limits the number of candidates. Without this, many selection methods will return an infinite
 * amount of candidates.
 * @param <T>
 *
 * @author Christian Skärby
 */
public class Limit<T> implements Selection<T> {

    private final LongSupplier limitSupplier;
    private final Selection<T> sourceSelection;

    public Limit(LongSupplier limitSupplier, Selection<T> sourceSelection) {
        this.limitSupplier = limitSupplier;
        this.sourceSelection = sourceSelection;
    }

    @Override
    public Stream<T> selectCandiates(List<Map.Entry<Double, T>> fitnessCandidates) {
        return sourceSelection.selectCandiates(fitnessCandidates).limit(limitSupplier.getAsLong());
    }

    /**
     * Convenience class to set a total limit among several {@link Selection}s
     */
    public static class FixedTotalLimit {
        private long remaining;

        public FixedTotalLimit(long fixedLimit) {
            this.remaining = fixedLimit;
        }

        /**
         * Limit the given {@link Selection}
         * @param limit Max number of candidates to select
         * @param toLimit {@link Selection} to limit
         * @return A limited {@link Selection}
         */
        public <T> Selection<T> limit(long limit, Selection<T> toLimit) {
            remaining -= limit;
            if(remaining < 0) {
                throw new IllegalArgumentException("Total limit exceeded!!");
            }
            return new Limit<>(() -> limit, toLimit);
        }

        /**
         * Give the remaining limit to the given {@link Selection}
         * @param toLimit {@link Selection} to limit
         * @return A limited {@link Selection}
         */
        public <T> Selection<T> last(Selection<T> toLimit) {
            final long limit = remaining;
            remaining = 0;
            return new Limit<>(() -> limit, toLimit);
        }

    }
}
