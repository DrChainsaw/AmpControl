package ampControl.audio.processing;

import org.jetbrains.annotations.Nullable;

import java.util.function.Supplier;

/**
 * Creates a {@link Supplier} for {@link ProcessingResult.Processing} based on a name string. Used for persistence since
 * all preprocessing needs be recreated when restoring a saved model.
 * <br><br>
 * This class, like much else in this project is the result of passing time while training.
 * TODO Probably would have been wiser to have a serializable factory instead...
 *
 * @author Christian Skärby
 */
public class SupplierFactory {

    private final double samplingFreq;
    private final static String prefix = "_sgpp_";

    public SupplierFactory(double samplingFreq) {
        this.samplingFreq = samplingFreq;
    }

    public Supplier<ProcessingResult.Processing> get(String nameStr) {
        // System.out.println("Post proc factory got " + nameStr);

        String preForkStr = nameStr;
        //  System.out.println("get nameStr: " + nameStr);
        if (nameStr.matches(matchStr(".*" + Fork.matchStrStatic()))) {
            Supplier<ProcessingResult.Processing> forkSupplier;
            String[] forkStrs = Fork.splitFirst(nameStr);
            //   System.out.println("len: " + forkStrs.length);
            preForkStr = forkStrs[0];
            if (forkStrs.length == 2) {
                final String forkStr = forkStrs[1];
                //   System.out.println("Got next fork: " + forkStr);
                final String[] endSplit = Fork.splitEnd(forkStr);
                //   System.out.println("endSplit: " + Arrays.toString(endSplit));
                final String pathStr = endSplit[0];
                //   System.out.println("Got next krof: " + pathStr);
                final String[] paths = Fork.splitMid(pathStr);
                //  System.out.println("get paths: " + paths[0] + " " + paths[1]);
                // System.out.println("ask for: " + prefix() + paths[0]);
                //  System.out.println("ask for: " + prefix() + paths[1]);
                final Supplier<ProcessingResult.Processing> finalFirst = get(prefix() + paths[0]);
                final Supplier<ProcessingResult.Processing> finalSecond = get(prefix() + paths[1]);
                forkSupplier = () -> new Fork(finalFirst.get(), finalSecond.get());

                if (endSplit.length == 2 && endSplit[1].contains(Pipe.nameStatic())) {
                    //    System.out.println("Append pipes " + endSplit[1]);
                    forkSupplier = getPipedSupplier(endSplit[1], forkSupplier, null);
                }

                if (preForkStr.replace(prefix(), "").isEmpty()) {
                    return forkSupplier;
                }
                //  System.out.println("Prepend pipes " + preForkStr + " fork: " + forkSupplier.get().name());
                return getPipedSupplier(preForkStr, null, forkSupplier);
            }
        }

        if (preForkStr.matches(matchStr(".*" + Pipe.nameStatic()))) {
            return getPipedSupplier(preForkStr, null, null);
        }

        if (preForkStr.matches(matchStr(NoProcessing.nameStatic()))) {
            return NoProcessing::new;
        }

        if (preForkStr.matches(matchStr(UnitMaxZeroMin.nameStatic()))) {
            return UnitMaxZeroMin::new;
        }


        if (preForkStr.matches(matchStr(LogScale.nameStatic()))) {
            return LogScale::new;
        }

        if (preForkStr.matches(matchStr(UnitStdZeroMean.nameStatic()))) {
            return UnitStdZeroMean::new;
        }

        if (preForkStr.matches(matchStr(Mfsc.nameStatic()))) {
            return () -> new Mfsc(samplingFreq);
        }

        if (preForkStr.matches(matchStr(Dct.nameStatic()))) {
            return Dct::new;
        }

        if (preForkStr.matches(matchStr(Spectrogram.nameStatic()))) {
            final String specParStr = preForkStr;
            return () -> new Spectrogram(specParStr);
        }

        if (preForkStr.matches(matchStr(ZeroMean.nameStatic()))) {
            return ZeroMean::new;
        }

        if (preForkStr.matches(matchStr(Log10.nameStatic()))) {
            return Log10::new;
        }

        return UnitMaxZeroMean::new;
    }

    @Nullable
    private Supplier<ProcessingResult.Processing> getPipedSupplier(
            String nameStr,
            Supplier<ProcessingResult.Processing> first,
            Supplier<ProcessingResult.Processing> last) {
        Supplier<ProcessingResult.Processing> ret = null;
        for (String pipeStr : nameStr.split(Pipe.nameStatic())) {
            if (!pipeStr.isEmpty()) {
                if (first == null) {
                    // System.out.println("Ask first pipe: " + pipeStr);
                    ret = get(pipeStr);
                    first = ret;
                } else {
                    // System.out.println("do pipe for: " + first.get().name() + " and " + pipeStr);
                    final Supplier<ProcessingResult.Processing> finalFirst = first;
                    final Supplier<ProcessingResult.Processing> finalSecond = get(prefix() + pipeStr);
                    // System.out.println("Got second: " + finalSecond.get().name());
                    ret = () -> new Pipe(finalFirst.get(), finalSecond.get());
                    first = ret;
                }
            }
        }

        if (last != null) {
            final Supplier<ProcessingResult.Processing> finalFirst = ret;
            final Supplier<ProcessingResult.Processing> finalSecond = last;
            ret = () -> new Pipe(finalFirst.get(), finalSecond.get());
        }
        return ret;
    }

    public static String prefix() {
        return prefix;
    }

    private static String matchStr(String aStr) {
        return ".*" + prefix() + aStr + ".*";
    }

    public static void main(String[] args) {
        //System.out.println("Got: " + new SupplierFactory(44100).get("_sgpp_fork_uszm_split_mfcc_krof_aswa_gtht_vrvrre33").get().name());
        //System.out.println("Got: " + new SupplierFactory(44100).get("_sgpp_fork_uszm_pipe_norm_split_mfcc_pipe_lgsc_krof_aswa_gtht_vrvrre33").get().name());
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

        String str = prefix() + pp.name();//"_sgpp_mfsc_pipe_fork_uszm_pipe_fork_norm_split_mfsc_krof_pipe_mfsc_pipe_mfcc_split_norm_pipe_mfsc_krof_aswa_gtht_vrvrre33";
        //String str = "_sgpp_fork_fork_fork_norm_split_mfcc_krof__split_mfcc_krof__split_uszm_krof_aswa_gtht_vrvrre33";
        Supplier<ProcessingResult.Processing> pps = new SupplierFactory(44100).get(str);
        System.out.println("Created: " + prefix() + pp.name());
        System.out.println("Wanted:  " + str);
        System.out.println("Got:     " + prefix() + pps.get().name());

        //Lets try something I can imagine might happen....
        pp = new Fork(
                new Pipe(
                        new Mfsc(10),
                        new UnitMaxZeroMin()
                ),
                new Pipe(
                        new Spectrogram(512, 32),
                        new LogScale()
                )

        );
        pps = new SupplierFactory(44100).get(prefix() + pp.name());
        System.out.println("Created: " + pp.name());
        System.out.println("Got:     " + pps.get().name());

        //Or perhaps even...
        pp = new Fork(
                pp,
                new UnitMaxZeroMin()
        );
        pps = new SupplierFactory(44100).get(prefix() + pp.name());
        System.out.println("Created: " + pp.name());
        System.out.println("Got:     " + pps.get().name());

    }
}
