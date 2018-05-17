package ampcontrol.model.training.data.processing;

import org.junit.Test;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.junit.Assert.assertArrayEquals;

/**
 * Test cases for {@link RandomFileSupplier}
 *
 * @author Christian Skärby
 */
public class RandomFileSupplierTest {

    /**
     * Test that paths are provided in order according to provided {@link Random}.
     */
    @Test
    public void get() {
        final List<Path> fakeFiles = Arrays.asList(
                Paths.get("fefew"),
                Paths.get("tregg"),
                Paths.get("qwwrt"),
                Paths.get("htrrt"),
                Paths.get("rgrgr")
        );
        final int[] expectedSequence = IntStream.range(0,20).map(i -> i+3).map(i -> i % fakeFiles.size()).toArray();
        final Random mockRandom = new Random() {
            int cnt = 0;
            @Override
            public int nextInt(int bound) {
                return expectedSequence[cnt++];
            }
        };
        final Supplier<Path> fileSupplier = new RandomFileSupplier(mockRandom, fakeFiles);

        final int[] actualSequence = Stream.generate(fileSupplier)
                .limit(expectedSequence.length)
                .mapToInt(aPath -> fakeFiles.indexOf(aPath))
                .toArray();
        assertArrayEquals("Incorrect sequence!", expectedSequence,actualSequence);
    }
}