// The inlined worker bundle is imported as text (esbuild --loader:.txt=text).
declare module "*.txt" {
  const text: string;
  export default text;
}
