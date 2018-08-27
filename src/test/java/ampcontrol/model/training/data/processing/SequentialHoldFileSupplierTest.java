package ampcontrol.model.training.data.processing;

import ampcontrol.model.training.data.state.ResetableStateFactory;
import ampcontrol.model.training.data.state.SimpleStateFactory;
import org.junit.Test;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * Test cases for {@link SequentialHoldFileSupplier}
 *
 * @author Christian Skärby
 */
public class SequentialHoldFileSupplierTest {

    /**
     * Test that files are supplied in the expected sequence.
     */
    @Test
    public void get() {
        final int startInd = 1;
        final int nrToHold = 3;
        final List<Path> fakeFiles = Arrays.asList(
                Paths.get("fefew"),
                Paths.get("tregg")
        );
        final List<Path> expected = Stream.concat(
                Collections.nCopies(nrToHold, fakeFiles.get(startInd)).stream(),
                Collections.nCopies(nrToHold, fakeFiles.get(startInd - 1)).stream()
        ).collect(Collectors.toList());

        final Supplier<Path> fileSupplier = new SequentialHoldFileSupplier(fakeFiles, nrToHold, startInd, new SimpleStateFactory(0));

        final List<Path> actual = Stream.generate(fileSupplier).limit(expected.size()).collect(Collectors.toList());
        assertEquals("Incorrect sequence!", expected, actual);
    }

    /**
     * Test that the exact same sequence can be reproduced if state is restored
     */
    @Test
    public void restoreState() {
        final int nrToHold = 3;
        final int testSize = 103;
        final List<Path> fakeFiles = IntStream.range(0, 32).mapToObj(i -> Paths.get("dummy" + i)).collect(Collectors.toList());
        final ResetableStateFactory controller = new ResetableStateFactory(567);

        final Supplier<Path> fileSupplier = new SequentialHoldFileSupplier(fakeFiles, nrToHold, controller);
        final List<Path> prev = Stream.generate(fileSupplier).limit(testSize).collect(Collectors.toList());
        controller.storeCurrentState();

        final List<Path> expected = Stream.generate(fileSupplier).limit(testSize).collect(Collectors.toList());
        assertNotEquals("Expected a new sequence!", prev, expected);
        controller.restorePreviousState();
        final List<Path> actual = Stream.generate(fileSupplier).limit(testSize).collect(Collectors.toList());
        assertEquals("Incorrect sequence!", expected, actual);
    }
}