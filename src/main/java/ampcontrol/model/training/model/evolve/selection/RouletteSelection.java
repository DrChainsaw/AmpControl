package ampcontrol.model.training.model.evolve.selection;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.function.DoubleSupplier;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * Roulette wheel selection, a.k.a Fitness Proportional Selection. Candidates as selected with a probability which is
 * inversely proportional to their fitness, i.e better fitness -> higher chance to be selected.
 * @param <T>
 *
 * @author Christian Skärby
 */
public final class RouletteSelection<T> implements Selection<T> {

    private static final Logger log = LoggerFactory.getLogger(RouletteSelection.class);

    private final DoubleSupplier rng;

    public RouletteSelection(DoubleSupplier rng) {
        this.rng = rng;
    }

    @Override
    public Stream<T> selectCandiates(List<Map.Entry<Double, T>> fitnessCandidates) {

        final double fitnessSum = fitnessCandidates.stream().mapToDouble(d -> 1 / d.getKey()).sum();
        return DoubleStream.generate(rng)
                .peek(this::assertRng)
                .map(uniform -> fitnessSum * uniform)
                .mapToObj(fitnessSelection -> selectCand(fitnessSelection, fitnessCandidates));
    }

    private T selectCand(double fitnessSelection, List<Map.Entry<Double, T>> fitnessCandidates) {
        double accVal = 0;
        for (Map.Entry<Double, T> entry : fitnessCandidates) {
            accVal += 1 / entry.getKey();
            if (accVal > fitnessSelection) {
                log.info("Selected cand " + fitnessCandidates.indexOf(entry) + " with fitness " + entry.getKey());
                return entry.getValue();
            }
        }
        throw new IllegalStateException("No cand found for selection " + fitnessSelection + "!");
    }

    private void assertRng(double draw) {
        if(!(draw >= 0 && draw < 1)) {
            throw new IllegalArgumentException("Random numbers must be on interval 0 <= x < 1! Was: " + draw);
        }
    }
}
