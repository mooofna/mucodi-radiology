// Minimal types for @niivue/dcm2niix (DICOM to NIfTI in WebAssembly, in its own worker).
declare module '@niivue/dcm2niix' {
  interface Processor {
    run(): Promise<File[]>;
    z(value: 'y' | 'n' | 'o' | 'i'): Processor;
    b(value: 'y' | 'n' | 'o'): Processor;
  }
  export class Dcm2niix {
    init(): Promise<boolean>;
    input(files: ArrayLike<File>): Processor;
    inputFromWebkitDirectory(files: ArrayLike<File>): Processor;
  }
}
