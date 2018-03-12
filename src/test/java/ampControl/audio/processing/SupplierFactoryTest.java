package ampControl.audio.processing;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Test cases for {@link SupplierFactory}.
 *
 */
public class SupplierFactoryTest {

    /**
     * Tests that a created {@link ProcessingResult.Processing} can be recreated
     */
    @Test
    public void get() {
        ProcessingResult.Processing pp = new Pipe(
                new Pipe(
                        new Mfsc(100),
                        new Dct()),
                new Fork(
                        new Pipe(
                                new UnitStdZeroMean(),
                                new Pipe(
                                        new Fork(
                                                new UnitMaxZeroMin(),
                                                new Mfsc(100)),
                                        new Pipe(
                                                new Mfsc(10),
                                                new Mfsc(10)
                                        )
                                )
                        ),
                        new Pipe(
                                new UnitMaxZeroMin(),
                                new Mfsc(10)
                        )));

        String str = "weewf21_23fd_" + SupplierFactory.prefix() + pp.name() + "_f5re5r7_hy6t8juy45";
        ProcessingResult.Processing pps = new SupplierFactory(44100).get(str).get();
        assertEquals("Processing was not restored correctly!", pp.name(), pps.name());
    }
}