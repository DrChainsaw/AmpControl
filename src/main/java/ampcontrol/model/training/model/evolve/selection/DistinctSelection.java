package ampcontrol.model.training.model.evolve.selection;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Stream;

/**
 * Removes duplicate candidates after selection. Helps maintaining genetic diversity when mutation rate is low.
 * Candidates must be mapped to a uniqueness identifier through a provided function.
 *
 * <>br<br>
 * Builder can create child {@link Selection}s which share the same state w.r.t. seen candidates which is useful if
 * a {@link CompoundSelection} is used. For this to work properly, the "main" {@link DistinctSelection} must be placed
 * above the {@link CompoundSelection} in the hierarchy so that state is cleared correctly before each selection.
 *
 * @param <T> Type of candidates
 * @param <V> Type of uniqueness identifier
 *
 * @author Christian Skärby
 */
public class DistinctSelection<T, V> implements Selection<T>  {

    private static final Logger log = LoggerFactory.getLogger(DistinctSelection.class);

    private final Set<V> seenCandidates;
    private final FilterUnseen<T,V> filter;
    private final Selection<T> source;

    private static final class FilterUnseen<T, V> implements Predicate<T> {

        private final Function<T, V> mapping;
        private final Set<V> seenCandidates;

        private FilterUnseen(Function<T, V> mapping, Set<V> seenCandidates) {
            this.mapping = mapping;
            this.seenCandidates = seenCandidates;
        }

        @Override
        public synchronized boolean test(T t) {
            if(seenCandidates.add(mapping.apply(t))) {
                return true;
            }
            log.info("Remove duplicate: " + t);
            return false;
        }
    }

    /**
     * Create a new Builder instance
     * @param uniquenessMapping Maps candidates to a uniqueness identifier
     * @param <T> Type of candidates
     * @param <V> Type of candidate uniqueness identifier
     * @return a new Builder instance
     */
    public static <T, V> Builder<T, V> builder(Function<T, V> uniquenessMapping) {
        return new Builder<>(uniquenessMapping);
    }

    public DistinctSelection(Function<T, V> uniquenessMapping, Selection<T> source) {
        this(Collections.emptySet(), uniquenessMapping, source);
    }


    public DistinctSelection(Set<V> seenCandidates, Function<T, V> uniquenessMapping, Selection<T> source) {
        this.source = source;
        this.seenCandidates = seenCandidates;
        this.filter = new FilterUnseen<>(uniquenessMapping, new HashSet<>());
    }

    @Override
    public Stream<T> selectCandiates(List<Map.Entry<Double, T>> fitnessCandidates) {
        seenCandidates.clear();
        filter.seenCandidates.clear();
        return source.selectCandiates(fitnessCandidates).filter(filter);
    }

    /**
     * Builder for this class.
     */
    public static final class Builder<T, V> {

        // Probably many things in this package which won't work for parallel streams, but just for good measure...
        private final Set<V> seenCandidates = Collections.synchronizedSet(new HashSet<>());
        private Selection<T> source;
        private final Function<T, V> uniquenessMapping;


        /**
         * Use static method in main class to invoke
         */
        private Builder(Function<T, V> uniquenessMapping) {
            this.uniquenessMapping = uniquenessMapping;
        }

        /**
         * Sets the source selection
         * @param source Selection used to get candidates
         * @return the Builder for fluent API
         */
        public Builder<T, V> source(Selection<T> source) {
            this.source = source;
            return this;
        }

        /**
         * Makes the given {@link Selection} only return candidates not seen by any other {@link Selection} (including
         * itself) added through this method. Useful in combination with a {@link CompoundSelection} as each component
         * can be configured to produce candidates which are distinct across all components.
         * <br><br>
         *     Not that {@link Selection}s may be added even after the {@link DistinctSelection} has been built.
         * @param source {@link Selection} to make distinct
         * @return A distinct version of source
         */
        public Selection<T> distinct(Selection<T> source) {
            final Predicate<T> filterUnseen = new FilterUnseen<>(uniquenessMapping, seenCandidates);
            return fitnessCandidates -> source.selectCandiates(fitnessCandidates)
                    .filter(filterUnseen);
        }

        /**
         * Construct a new {@link DistinctSelection} instance
         * @return a new {@link DistinctSelection} instance
         */
        public DistinctSelection<T,V> build() {
            return new DistinctSelection<>(seenCandidates, uniquenessMapping, source);
        }

    }

}
