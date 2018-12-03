package ampcontrol.model.training.model.evolve.selection;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.DoubleSupplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link RouletteSelection}
 *
 * @author Christian Skärby
 */
public class RouletteSelectionTest {

    /**
     * Test that the expected candiates are selected
     */
    @Test
    public void selectCandiates() {
        final List<Map.Entry<Double, MockEvolvingItem>> fitnessCands = MockEvolvingItem.createFitnessCands(9, i-> i+1);

        final double sumFitness = fitnessCands.stream().mapToDouble(Map.Entry::getKey).map(d -> 1/d).sum();

        final List<Integer> expectedCands = Arrays.asList(0, 1, 2, 3, 5, 7, 0, fitnessCands.size()-1, 3);
        final DoubleSupplier mockRng = new DoubleSupplier() {
            int cnt = 0;

            @Override
            public double getAsDouble() {
                return IntStream.range(0, expectedCands.get(cnt++ % expectedCands.size()))
                        .mapToDouble(i -> fitnessCands.get(i).getKey())
                        .map(d -> 1/d + 1e-4) // for numerical stability
                        .sum() / sumFitness;
            }
        };

        final Selection<MockEvolvingItem> selection = new RouletteSelection<>(mockRng);

        assertEquals("Incorrect selection!",
                expectedCands.stream()
                        .map(fitnessCands::get)
                        .map(Map.Entry::getValue)
                        .collect(Collectors.toList()),
                selection.selectCandiates(fitnessCands)
                        .limit(expectedCands.size())
                        .collect(Collectors.toList()));
    }

    /**
     * Test that candidates are selected with the expected frequency when {@link java.util.Random} is used for selection
     */
    @Test
    public void expectedFrequency() {
        final double[] freqs = {0.5, 0.3, 0.15, 0.05};
        final List<Map.Entry<Double, MockEvolvingItem>> fitnessCands = MockEvolvingItem.createFitnessCands(freqs.length, i -> 1 / freqs[i]);

        final Random rng = new Random(666);
        final Selection<MockEvolvingItem> selection = new RouletteSelection<>(rng::nextDouble);
        final double[] pdf = new double[freqs.length];

        final long nrToDraw = 100000;
        selection.selectCandiates(fitnessCands)
                //.parallel()
                .limit(nrToDraw)
                .forEach(cand -> {synchronized(pdf) {pdf[Integer.parseInt(cand.toString())]++;}});

        IntStream.range(0, freqs.length).forEach(i ->
                assertEquals("Incorrect frequency: ", freqs[i], pdf[i] / nrToDraw, 1e-2 ));
    }

}