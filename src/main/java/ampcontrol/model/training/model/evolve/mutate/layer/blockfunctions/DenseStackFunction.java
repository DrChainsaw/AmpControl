package ampcontrol.model.training.model.evolve.mutate.layer.blockfunctions;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.graph.DenseStack;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * Creates a {@link DenseStack} using {@link LayerBlockConfig}s from a source function. Number of stacks is selected by
 * a provided Function. Which numbers of stacks for a given nOut are possible is limited since the condition
 * nOut == nOutPerBlockInStack * nrofBlocksToStack must be true.
 *
 * @author Christian Skärby
 */
public class DenseStackFunction implements Function<Long, LayerBlockConfig> {

    private final Function<List<Long>, Integer> nrofStacksSelector;
    private final Function<Long, LayerBlockConfig> source;

    public DenseStackFunction(
            Function<List<Long>, Integer> nrofStacksSelector,
            Function<Long, LayerBlockConfig> source) {
        this.nrofStacksSelector = nrofStacksSelector;
        this.source = source;
    }

    @Override
    public LayerBlockConfig apply(Long nOut) {
        final List<Long> factorization = primeFactors(nOut);
        if (factorization.size() == 1) {
            // Prime number, no dense block possible so that nOut of block == nOut
            return source.apply(nOut);
        }

        final int nrofStackSelection = nrofStacksSelector.apply(factorization);
        final int stackSize = factorization.remove(nrofStackSelection).intValue();
        final long nOutOfComponent = factorization.stream().reduce(1L, (l1, l2) -> l1 * l2);
        return new DenseStack()
                .setBlockToStack(source.apply(nOutOfComponent))
                .setNrofStacks(stackSize);
    }


    /**
     * Copy pasted from http://www.vogella.com/tutorials/JavaAlgorithmsPrimeFactorization/article.html
     *
     * @param number Number to factorize
     * @return List of primes
     */
    private static List<Long> primeFactors(long number) {
        long n = number;
        List<Long> factors = new ArrayList<>();
        for (long i = 2; i <= n / i; i++) {
            while (n % i == 0) {
                factors.add(i);
                n /= i;
            }
        }
        if (n > 1) {
            factors.add(n);
        }
        return factors;
    }
}
