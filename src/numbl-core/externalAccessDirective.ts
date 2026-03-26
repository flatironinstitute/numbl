export interface ExternalAccessDirectives {
  /** Variable names declared at file/script scope (outside any function) */
  fileScope: Set<string>;
  /** Function name -> variable names declared within that function */
  functionScope: Map<string, Set<string>>;
}
