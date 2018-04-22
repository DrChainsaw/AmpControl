package ampControl.audio.processing;

import org.jetbrains.annotations.Nullable;

/**
 * Creates a {@link ProcessingResult.Factory} based on a name string. Used for persistence since
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

    public ProcessingResult.Factory get(String nameStr) {
        // System.out.println("Post proc factory got " + nameStr);

        String preForkStr = nameStr;
        //  System.out.println("get nameStr: " + nameStr);
        if (nameStr.matches(matchStr(".*" + Fork.matchStrStatic()))) {
            ProcessingResult.Factory forkSupplier;
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
                final ProcessingResult.Factory finalFirst = get(prefix() + paths[0]);
                final ProcessingResult.Factory finalSecond = get(prefix() + paths[1]);
                forkSupplier = new Fork(finalFirst, finalSecond);

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
            return new NoProcessing();
        }

        if (preForkStr.matches(matchStr(UnitMaxZeroMin.nameStatic()))) {
            return new UnitMaxZeroMin();
        }


        if (preForkStr.matches(matchStr(LogScale.nameStatic()))) {
            return new LogScale();
        }

        if (preForkStr.matches(matchStr(UnitStdZeroMean.nameStatic()))) {
            return new UnitStdZeroMean();
        }

        if (preForkStr.matches(matchStr(Mfsc.nameStatic()))) {
            return new Mfsc(samplingFreq);
        }

        if (preForkStr.matches(matchStr(Dct.nameStatic()))) {
            return new Dct();
        }

        if (preForkStr.matches(matchStr(Spectrogram.nameStatic()))) {
            return new Spectrogram(preForkStr);
        }

        if (preForkStr.matches(matchStr(ZeroMean.nameStatic()))) {
            return new ZeroMean();
        }

        if (preForkStr.matches(matchStr(Log10.nameStatic()))) {
            return new Log10();
        }

        return new UnitMaxZeroMean();
    }

    @Nullable
    private ProcessingResult.Factory getPipedSupplier(
            String nameStr,
            ProcessingResult.Factory first,
            ProcessingResult.Factory last) {
        ProcessingResult.Factory ret = null;
        for (String pipeStr : nameStr.split(Pipe.nameStatic())) {
            if (!pipeStr.isEmpty()) {
                if (first == null) {
                    // System.out.println("Ask first pipe: " + pipeStr);
                    ret = get(pipeStr);
                    first = ret;
                } else {
                    // System.out.println("do pipe for: " + first.get().name() + " and " + pipeStr);
                    final ProcessingResult.Factory second = get(prefix() + pipeStr);
                    // System.out.println("Got second: " + finalSecond.get().name());
                    ret = new Pipe(first, second);
                    first = ret;
                }
            }
        }

        if (last != null) {
            ret = new Pipe(ret, last);
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
        ProcessingResult.Factory pp = new Pipe(
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
        ProcessingResult.Factory pps = new SupplierFactory(44100).get(str);
        System.out.println("Created: " + prefix() + pp.name());
        System.out.println("Wanted:  " + str);
        System.out.println("Got:     " + prefix() + pps.name());

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
        System.out.println("Got:     " + pps.name());

        //Or perhaps even...
        pp = new Fork(
                pp,
                new UnitMaxZeroMin()
        );
        pps = new SupplierFactory(44100).get(prefix() + pp.name());
        System.out.println("Created: " + pp.name());
        System.out.println("Got:     " + pps.name());

    }
}
