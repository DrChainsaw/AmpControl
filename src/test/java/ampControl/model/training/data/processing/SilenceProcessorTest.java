package ampControl.model.training.data.processing;

import ampControl.audio.processing.NoProcessing;
import org.junit.Test;

import java.util.stream.DoubleStream;

import static org.junit.Assert.*;

/**
 * Test cases for {@link SilenceProcessor}
 *
 * @author Christian Skärby
 */
public class SilenceProcessorTest {

    /**
     * Test that SilenceProcessor can perform the monumental task of creating an array of zeroes
     */
    @Test
    public void getResult() {
        final int size = 7;
        double[][] expected = new double[][] {DoubleStream.generate(()->0).limit(size).toArray()};
        final AudioProcessor sp = new SilenceProcessor(7, ()-> new NoProcessing());
        assertArrayEquals("Inorrect result!", expected, sp.getResult().get().get(0));
    }
}