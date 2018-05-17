package ampcontrol.amp.midi;

import ampcontrol.amp.midi.program.PodXtProgramChange;
import com.beust.jcommander.JCommander;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.sound.midi.ShortMessage;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link ProbabilitiesToMidiProgramChange}.
 */
public class ProbabilitiesToMidiProgramChangeTest {

    private final static String labelToThreshPar = "-ltt ";
    private final static String labelMaskPar = "-lm ";
    private final static String switchMomThreshPar = "-st ";
    private final static String updateProhibitPar = "-upt ";

    private final static List<ShortMessage> noOutput = Collections.emptyList();

    @Test
    public void probabilitiesToMessageMapping() {
        final int midiChan = 7;
        final List<PodXtProgramChange> programChanges = Arrays.asList(PodXtProgramChange.A4, PodXtProgramChange.B17, PodXtProgramChange.C7);
        final List<Double> pThresholds = Arrays.asList(0.5, 0.7, 0.8);
        final int labelMask = 0;
        final int switchMomThresh = 3;
        final int updateProhibitTimeMs = 0;
        final String argStr =
                labelToThreshPar + IntStream.range(0, pThresholds.size()).mapToObj(lab -> lab + ":" + pThresholds.get(lab)).collect(Collectors.joining(",")) +
                " " + labelMaskPar + labelMask +
                " " + switchMomThreshPar + switchMomThresh +
                " " + updateProhibitPar + updateProhibitTimeMs;


        ProbabilitiesToMidiProgramChange programChangeMapping = new ProbabilitiesToMidiProgramChange(() -> midiChan);
        JCommander.newBuilder().addObject(programChangeMapping)
                .build()
                .parse(argStr.split(" "));

        Function<INDArray, List<ShortMessage>> probMapping = programChangeMapping.createProbabilitiesToMessageMapping(programChanges);

        final INDArray nothing = Nd4j.create(new double[]{0.333, 0.333, 0.333});
        final INDArray lab0 = Nd4j.create(new double[]{0.6, 0.2, 0.2});
        final INDArray lab1 = Nd4j.create(new double[]{0.0, 0.8, 0.2});
        final INDArray lab2 = Nd4j.create(new double[]{0.05, 0.1, 0.85});

        // No probability above threshold: No output
        assertEquals("No output expected!", noOutput, probMapping.apply(nothing));

        // Label is masked: No output
        for(int i = 0; i < 2*switchMomThresh; i++) {
            assertEquals("No output expected!", noOutput, probMapping.apply(lab0));
        }
        // Alternate labels: No output due to momentum filter
        for(int i = 0; i < 2*switchMomThresh; i++) {
            assertEquals("No output expected!", noOutput, probMapping.apply(lab1));
            assertEquals("No output expected!", noOutput, probMapping.apply(lab2));
        }
        for(int i = 0; i < switchMomThresh-1; i++) {
            assertEquals("No output expected!", noOutput, probMapping.apply(lab1));
        }
        // No probability above threshold: No output and does not reset momentum filter
        assertEquals("No output expected!", noOutput, probMapping.apply(nothing));

        final List<ShortMessage> resultExpectLab1 = probMapping.apply(lab1);
        assertEquals("Incorrect size!", 1, resultExpectLab1.size());
        assertEquals("Incorrect channel!", midiChan, resultExpectLab1.get(0).getChannel());
        assertEquals("Incorrect message!", programChanges.get(1).program(), resultExpectLab1.get(0).getData1());

        for(int i = 0; i < switchMomThresh-1; i++) {
            assertEquals("No output expected!", noOutput, probMapping.apply(lab2));
        }
        // No probability above threshold: No output and does not reset momentum filter
        assertEquals("No output expected!", noOutput, probMapping.apply(nothing));

        final List<ShortMessage> resultExpectLab2 = probMapping.apply(lab2);
        assertEquals("Incorrect size!", 1, resultExpectLab1.size());
        assertEquals("Incorrect channel!", midiChan,  resultExpectLab1.get(0).getChannel());
        assertEquals("Incorrect message!", programChanges.get(2).program(), resultExpectLab2.get(0).getData1());
    }
}