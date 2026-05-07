# Sparse Matrix Detailed Properties

Complete information extracted from SuiteSparse Collection pages.

**Total matrices**: 29

---

## 1. pdb1HYS

**Collection**: Williams  
**URL**: https://sparse.tamu.edu/Williams/pdb1HYS

### Properties

- **Name**: pdb1HYS
- **Group**: Williams
- **Matrix Id**: 2373
- **Num Rows**: 36,417
- **Num Cols**: 36,417
- **Nonzeros**: 4,344,765
- **Pattern Entries**: 4,344,765
- **Kind**: Weighted Undirected Graph
- **Symmetric**: Yes
- **Date**: 2008
- **Author**: S. G. Sarafianos et al
- **Editor**: S. Williams, L. Oliker, R. Vuduc, J. Shalf, K. Yelick, J. Demmel
- **Strongly Connect Components**: 1
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 100%
- **Cholesky Candidate**: yes
- **Positive Definite**: yes
- **Type**: real
- **Matrix Norm**: 3.523937e+02
- **Minimum Singular Value**: 9.970386e-10
- **Condition Number**: 3.534404e+11
- **Rank**: 36,411
- **Null Space Dimension**: 6
- **Full Numerical Rank?**: no
- **Download Singular Values**: MATLAB
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: Matrices used by S. Williams et al for sparse matrix multiplication on GPUs.   
                                                                               
14 matrices were used in the following paper:                                  
                                                                               
    S. Williams, L. Oliker, R. Vuduc, J. Shalf, K. Yelick, J. Demmel,          
    "Optimization of Sparse Matrix-Vector Multiplication on Emerging Multicore 
    Platforms", Parallel Computing Volume 35, Issue 3, March 2009, Pages       
    178-194.  Special issue on Revolutionary Technologies for Acceleration of  
    Emerging Petascale Applications.                                           
                                                                               
    https://hpcrd.lbl.gov/~swwilliams/research/papers/parco08_spmv.pdf         
    http://dx.doi.org/10.1016/j.parco.2008.12.006                              
                                                                               
This same set of 14 matrices was also used in a subsequent technical report by 
NVIDIA:                                                                        
                                                                               
    http://www.nvidia.com/object/nvidia_research_pub_001.html "Efficient Sparse
    Matrix-Vector Multiplication on CUDA" Nathan Bell and Michael Garland, in, 
    "NVIDIA Technical Report NVR-2008-004", December 2008                      
                                                                               
file            Name            dim*    nnz     description                    
                                                                               
dense2          Dense           2K      4.0M    dense matrix in sparse format  
pdb1HYS         Protein         36K     4.3M    protein data bank 1HYS         
consph          FEM/Spheres     83K     6.0M    FEM concentric spheres         
cant            FEM/Cantilever  62K     4.0M    FEM cantilever                 
pwtk            Wind Tunnel     218K    11.6M   pressurized wind tunnel        
rma10           FEM/Harbor      47K     2.37M   3D CFD of Charleston Harbor    
qcd5_4          QCD             49K     1.90M   quark propagators (QCD/LGT)    
shipsec1        FEM/Ship        141K    3.98M   FEM Ship section / detail      
mac_econ_fwd500 Economics       207K    1.27M   Macroeconomic model            
mc2depi         Epidemiology    526K    2.1M    2D Markov model of epidemic    
cop20k_A        FEM/Accelerator 121K    2.62M   Accelerator cavity design      
scircuit        Circuit         171K    959K    Motorola circuit simulation    
webbase-1M      webbase         1M      3.1M    Web connectivity matrix        
rail4284        LP              4Kx1.1M 11.3M   Railways set cover,            
                                                constraint matrix              
                                                                               
(*) the matrix is square if only one dimension listed.                         
                                                                               
Six of the matrices are nearly identical to the matrices already in the        
UF Collection.  They are thus not included in the UF Collection.  See          
the README.txt file for this collection for details.                           
   I presume the pdb1HYS matrix comes from this source:                        
                                                                               
   http://www.rcsb.org/pdb/explore.do?structureId=1HYS                         
   http://dx.doi.org/10.2210/pdb1hys/pdb                                       
   Crystal structure of HIV-1 reverse transcriptase in complex with a          
   polypurine tract RNA:DNA.                                                   
   Sarafianos, S.G., Das, K., Tantillo, C., Clark Jr., A.D., Ding,             
   J., Whitcomb, J.M., Boyer, P.L., Hughes, S.H., Arnold, E.                   
   Journal: (2001) EMBO J. 20: 1449-1461                                       
   PubMed: 11250910                                                            
   PubMedCentral: PMC145536                                                    
   DOI: 10.1093/emboj/20.6.1449                                                
   Search Related Articles in PubMed                                           
   PubMed Abstract:                                                            
   We have determined the 3.0 A resolution structure of wild-type HIV-1        
   reverse transcriptase in complex with an RNA:DNA oligonucleotide whose      
   sequence includes a purine-rich segment from the HIV-1 genome called the    
   polypurine tract (PPT). The PPT is resistant to ribonuclease... [ Read      
   More & Search PubMed Abstracts ] We have determined the 3.0 A resolution    
   structure of wild-type HIV-1 reverse transcriptase in complex with an       
   RNA:DNA oligonucleotide whose sequence includes a purine-rich segment from  
   the HIV-1 genome called the polypurine tract (PPT). The PPT is resistant    
   to ribonuclease H (RNase H) cleavage and is used as a primer for second     
   DNA strand synthesis.  The "RNase H primer grip", consisting of amino       
   acids that interact with the DNA primer strand, may contribute to RNase H   
   catalysis and cleavage specificity. Cleavage specificity is also            
   controlled by the width of the minor groove and the trajectory of the       
   RNA:DNA, both of which are sequence dependent. An unusual "unzipping" of 7  
   bp occurs in the adenine stretch of the PPT: an unpaired base on the        
   template strand takes the base pairing out of register and then, following  
   two offset base pairs, an unpaired base on the primer strand                
   re-establishes the normal register. The structural aberration extends to    
   the RNase H active site and may play a role in the resistance of PPT to     
   RNase H cleavage.

---

## 2. consph

**Collection**: Williams  
**URL**: https://sparse.tamu.edu/Williams/consph

### Properties

- **Name**: consph
- **Group**: Williams
- **Matrix Id**: 2374
- **Num Rows**: 83,334
- **Num Cols**: 83,334
- **Nonzeros**: 6,010,480
- **Pattern Entries**: 6,010,480
- **Kind**: 2D/3D Problem
- **Symmetric**: Yes
- **Date**: 2008
- **Author**: unknown
- **Editor**: S. Williams, L. Oliker, R. Vuduc, J. Shalf, K. Yelick, J. Demmel
- **Structural Rank**: 83,334
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 3,656
- **Strongly Connect Components**: 3,656
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 100%
- **Cholesky Candidate**: yes
- **Positive Definite**: yes
- **Type**: real
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: Matrices used by S. Williams et al for sparse matrix multiplication on GPUs.   
                                                                               
14 matrices were used in the following paper:                                  
                                                                               
    S. Williams, L. Oliker, R. Vuduc, J. Shalf, K. Yelick, J. Demmel,          
    "Optimization of Sparse Matrix-Vector Multiplication on Emerging Multicore 
    Platforms", Parallel Computing Volume 35, Issue 3, March 2009, Pages       
    178-194.  Special issue on Revolutionary Technologies for Acceleration of  
    Emerging Petascale Applications.                                           
                                                                               
    https://hpcrd.lbl.gov/~swwilliams/research/papers/parco08_spmv.pdf         
    http://dx.doi.org/10.1016/j.parco.2008.12.006                              
                                                                               
This same set of 14 matrices was also used in a subsequent technical report by 
NVIDIA:                                                                        
                                                                               
    http://www.nvidia.com/object/nvidia_research_pub_001.html "Efficient Sparse
    Matrix-Vector Multiplication on CUDA" Nathan Bell and Michael Garland, in, 
    "NVIDIA Technical Report NVR-2008-004", December 2008                      
                                                                               
file            Name            dim*    nnz     description                    
                                                                               
dense2          Dense           2K      4.0M    dense matrix in sparse format  
pdb1HYS         Protein         36K     4.3M    protein data bank 1HYS         
consph          FEM/Spheres     83K     6.0M    FEM concentric spheres         
cant            FEM/Cantilever  62K     4.0M    FEM cantilever                 
pwtk            Wind Tunnel     218K    11.6M   pressurized wind tunnel        
rma10           FEM/Harbor      47K     2.37M   3D CFD of Charleston Harbor    
qcd5_4          QCD             49K     1.90M   quark propagators (QCD/LGT)    
shipsec1        FEM/Ship        141K    3.98M   FEM Ship section / detail      
mac_econ_fwd500 Economics       207K    1.27M   Macroeconomic model            
mc2depi         Epidemiology    526K    2.1M    2D Markov model of epidemic    
cop20k_A        FEM/Accelerator 121K    2.62M   Accelerator cavity design      
scircuit        Circuit         171K    959K    Motorola circuit simulation    
webbase-1M      webbase         1M      3.1M    Web connectivity matrix        
rail4284        LP              4Kx1.1M 11.3M   Railways set cover,            
                                                constraint matrix              
                                                                               
(*) the matrix is square if only one dimension listed.                         
                                                                               
Six of the matrices are nearly identical to the matrices already in the        
UF Collection.  They are thus not included in the UF Collection.  See          
the README.txt file for this collection for details.

---

## 3. mac_econ_fwd500

**Collection**: Williams  
**URL**: https://sparse.tamu.edu/Williams/mac_econ_fwd500

### Properties

- **Name**: mac_econ_fwd500
- **Group**: Williams
- **Matrix Id**: 2376
- **Num Rows**: 206,500
- **Num Cols**: 206,500
- **Nonzeros**: 1,273,389
- **Pattern Entries**: 1,273,389
- **Kind**: Economic Problem
- **Symmetric**: No
- **Date**: 2008
- **Author**: unknown
- **Editor**: S. Williams, L. Oliker, R. Vuduc, J. Shalf, K. Yelick, J. Demmel
- **Structural Rank**: 206,500
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 34
- **Strongly Connect Components**: 1
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 6%
- **Numeric Symmetry**: 0.6%
- **Cholesky Candidate**: no
- **Positive Definite**: no
- **Type**: real
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: Matrices used by S. Williams et al for sparse matrix multiplication on GPUs.   
                                                                               
14 matrices were used in the following paper:                                  
                                                                               
    S. Williams, L. Oliker, R. Vuduc, J. Shalf, K. Yelick, J. Demmel,          
    "Optimization of Sparse Matrix-Vector Multiplication on Emerging Multicore 
    Platforms", Parallel Computing Volume 35, Issue 3, March 2009, Pages       
    178-194.  Special issue on Revolutionary Technologies for Acceleration of  
    Emerging Petascale Applications.                                           
                                                                               
    https://hpcrd.lbl.gov/~swwilliams/research/papers/parco08_spmv.pdf         
    http://dx.doi.org/10.1016/j.parco.2008.12.006                              
                                                                               
This same set of 14 matrices was also used in a subsequent technical report by 
NVIDIA:                                                                        
                                                                               
    http://www.nvidia.com/object/nvidia_research_pub_001.html "Efficient Sparse
    Matrix-Vector Multiplication on CUDA" Nathan Bell and Michael Garland, in, 
    "NVIDIA Technical Report NVR-2008-004", December 2008                      
                                                                               
file            Name            dim*    nnz     description                    
                                                                               
dense2          Dense           2K      4.0M    dense matrix in sparse format  
pdb1HYS         Protein         36K     4.3M    protein data bank 1HYS         
consph          FEM/Spheres     83K     6.0M    FEM concentric spheres         
cant            FEM/Cantilever  62K     4.0M    FEM cantilever                 
pwtk            Wind Tunnel     218K    11.6M   pressurized wind tunnel        
rma10           FEM/Harbor      47K     2.37M   3D CFD of Charleston Harbor    
qcd5_4          QCD             49K     1.90M   quark propagators (QCD/LGT)    
shipsec1        FEM/Ship        141K    3.98M   FEM Ship section / detail      
mac_econ_fwd500 Economics       207K    1.27M   Macroeconomic model            
mc2depi         Epidemiology    526K    2.1M    2D Markov model of epidemic    
cop20k_A        FEM/Accelerator 121K    2.62M   Accelerator cavity design      
scircuit        Circuit         171K    959K    Motorola circuit simulation    
webbase-1M      webbase         1M      3.1M    Web connectivity matrix        
rail4284        LP              4Kx1.1M 11.3M   Railways set cover,            
                                                constraint matrix              
                                                                               
(*) the matrix is square if only one dimension listed.                         
                                                                               
Six of the matrices are nearly identical to the matrices already in the        
UF Collection.  They are thus not included in the UF Collection.  See          
the README.txt file for this collection for details.

---

## 4. mc2depi

**Collection**: Williams  
**URL**: https://sparse.tamu.edu/Williams/mc2depi

### Properties

- **Name**: mc2depi
- **Group**: Williams
- **Matrix Id**: 2377
- **Num Rows**: 525,825
- **Num Cols**: 525,825
- **Nonzeros**: 2,100,225
- **Pattern Entries**: 2,100,225
- **Kind**: 2D/3D Problem
- **Symmetric**: No
- **Date**: 2008
- **Author**: unknown
- **Editor**: S. Williams, L. Oliker, R. Vuduc, J. Shalf, K. Yelick, J. Demmel
- **Structural Rank**: 525,825
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 1
- **Strongly Connect Components**: 1
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 0%
- **Numeric Symmetry**: 0%
- **Cholesky Candidate**: no
- **Positive Definite**: no
- **Type**: integer
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: Matrices used by S. Williams et al for sparse matrix multiplication on GPUs.   
                                                                               
14 matrices were used in the following paper:                                  
                                                                               
    S. Williams, L. Oliker, R. Vuduc, J. Shalf, K. Yelick, J. Demmel,          
    "Optimization of Sparse Matrix-Vector Multiplication on Emerging Multicore 
    Platforms", Parallel Computing Volume 35, Issue 3, March 2009, Pages       
    178-194.  Special issue on Revolutionary Technologies for Acceleration of  
    Emerging Petascale Applications.                                           
                                                                               
    https://hpcrd.lbl.gov/~swwilliams/research/papers/parco08_spmv.pdf         
    http://dx.doi.org/10.1016/j.parco.2008.12.006                              
                                                                               
This same set of 14 matrices was also used in a subsequent technical report by 
NVIDIA:                                                                        
                                                                               
    http://www.nvidia.com/object/nvidia_research_pub_001.html "Efficient Sparse
    Matrix-Vector Multiplication on CUDA" Nathan Bell and Michael Garland, in, 
    "NVIDIA Technical Report NVR-2008-004", December 2008                      
                                                                               
file            Name            dim*    nnz     description                    
                                                                               
dense2          Dense           2K      4.0M    dense matrix in sparse format  
pdb1HYS         Protein         36K     4.3M    protein data bank 1HYS         
consph          FEM/Spheres     83K     6.0M    FEM concentric spheres         
cant            FEM/Cantilever  62K     4.0M    FEM cantilever                 
pwtk            Wind Tunnel     218K    11.6M   pressurized wind tunnel        
rma10           FEM/Harbor      47K     2.37M   3D CFD of Charleston Harbor    
qcd5_4          QCD             49K     1.90M   quark propagators (QCD/LGT)    
shipsec1        FEM/Ship        141K    3.98M   FEM Ship section / detail      
mac_econ_fwd500 Economics       207K    1.27M   Macroeconomic model            
mc2depi         Epidemiology    526K    2.1M    2D Markov model of epidemic    
cop20k_A        FEM/Accelerator 121K    2.62M   Accelerator cavity design      
scircuit        Circuit         171K    959K    Motorola circuit simulation    
webbase-1M      webbase         1M      3.1M    Web connectivity matrix        
rail4284        LP              4Kx1.1M 11.3M   Railways set cover,            
                                                constraint matrix              
                                                                               
(*) the matrix is square if only one dimension listed.                         
                                                                               
Six of the matrices are nearly identical to the matrices already in the        
UF Collection.  They are thus not included in the UF Collection.  See          
the README.txt file for this collection for details.

---

## 5. cop20k_A

**Collection**: Williams  
**URL**: https://sparse.tamu.edu/Williams/cop20k_A

### Properties

- **Name**: cop20k_A
- **Group**: Williams
- **Matrix Id**: 2378
- **Num Rows**: 121,192
- **Num Cols**: 121,192
- **Nonzeros**: 2,624,331
- **Pattern Entries**: 2,624,331
- **Kind**: 2D/3D Problem
- **Symmetric**: Yes
- **Date**: 2008
- **Author**: unknown
- **Editor**: S. Williams, L. Oliker, R. Vuduc, J. Shalf, K. Yelick, J. Demmel
- **Structural Rank**: 99,843
- **Structural Rank Full**: false
- **Num Dmperm Blocks**: 3
- **Strongly Connect Components**: 21,350
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 100%
- **Cholesky Candidate**: no
- **Positive Definite**: no
- **Type**: real
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: Matrices used by S. Williams et al for sparse matrix multiplication on GPUs.   
                                                                               
14 matrices were used in the following paper:                                  
                                                                               
    S. Williams, L. Oliker, R. Vuduc, J. Shalf, K. Yelick, J. Demmel,          
    "Optimization of Sparse Matrix-Vector Multiplication on Emerging Multicore 
    Platforms", Parallel Computing Volume 35, Issue 3, March 2009, Pages       
    178-194.  Special issue on Revolutionary Technologies for Acceleration of  
    Emerging Petascale Applications.                                           
                                                                               
    https://hpcrd.lbl.gov/~swwilliams/research/papers/parco08_spmv.pdf         
    http://dx.doi.org/10.1016/j.parco.2008.12.006                              
                                                                               
This same set of 14 matrices was also used in a subsequent technical report by 
NVIDIA:                                                                        
                                                                               
    http://www.nvidia.com/object/nvidia_research_pub_001.html "Efficient Sparse
    Matrix-Vector Multiplication on CUDA" Nathan Bell and Michael Garland, in, 
    "NVIDIA Technical Report NVR-2008-004", December 2008                      
                                                                               
file            Name            dim*    nnz     description                    
                                                                               
dense2          Dense           2K      4.0M    dense matrix in sparse format  
pdb1HYS         Protein         36K     4.3M    protein data bank 1HYS         
consph          FEM/Spheres     83K     6.0M    FEM concentric spheres         
cant            FEM/Cantilever  62K     4.0M    FEM cantilever                 
pwtk            Wind Tunnel     218K    11.6M   pressurized wind tunnel        
rma10           FEM/Harbor      47K     2.37M   3D CFD of Charleston Harbor    
qcd5_4          QCD             49K     1.90M   quark propagators (QCD/LGT)    
shipsec1        FEM/Ship        141K    3.98M   FEM Ship section / detail      
mac_econ_fwd500 Economics       207K    1.27M   Macroeconomic model            
mc2depi         Epidemiology    526K    2.1M    2D Markov model of epidemic    
cop20k_A        FEM/Accelerator 121K    2.62M   Accelerator cavity design      
scircuit        Circuit         171K    959K    Motorola circuit simulation    
webbase-1M      webbase         1M      3.1M    Web connectivity matrix        
rail4284        LP              4Kx1.1M 11.3M   Railways set cover,            
                                                constraint matrix              
                                                                               
(*) the matrix is square if only one dimension listed.                         
                                                                               
Six of the matrices are nearly identical to the matrices already in the        
UF Collection.  They are thus not included in the UF Collection.  See          
the README.txt file for this collection for details.

---

## 6. circuit5M

**Collection**: Freescale  
**URL**: https://sparse.tamu.edu/Freescale/circuit5M

### Properties

- **Name**: circuit5M
- **Group**: Freescale
- **Matrix Id**: 2276
- **Num Rows**: 5,558,326
- **Num Cols**: 5,558,326
- **Nonzeros**: 59,524,291
- **Pattern Entries**: 59,524,291
- **Kind**: Circuit Simulation Problem
- **Symmetric**: No
- **Date**: 2010
- **Author**: K. Gullapalli
- **Editor**: T. Davis
- **Structural Rank**: 5,558,326
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 1,094
- **Strongly Connect Components**: 647
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 42%
- **Cholesky Candidate**: no
- **Positive Definite**: no
- **Type**: real
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: Large circuit from Kiran Gullapalli, Freescale Semiconductor
Little fill-in during factorization, if ordered properly..

---

## 7. webbase-2001

**Collection**: LAW  
**URL**: https://sparse.tamu.edu/LAW/webbase-2001

### Properties

- **Name**: webbase-2001
- **Group**: LAW
- **Matrix Id**: 2449
- **Num Rows**: 118,142,155
- **Num Cols**: 118,142,155
- **Nonzeros**: 1,019,903,190
- **Pattern Entries**: 1,019,903,190
- **Kind**: Directed Graph
- **Symmetric**: No
- **Date**: 2004
- **Author**: Laboratory for Web Algorithmics (LAW), Universita degli Studi di Milano, http://law.di.unimi.it/index.php
- **Editor**: Laboratory for Web Algorithmics (LAW), Universita degli Studi di Milano, http://law.di.unimi.it/index.php
- **Strongly Connect Components**: 41,126,852
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 27.8%
- **Numeric Symmetry**: 27.8%
- **Cholesky Candidate**: no
- **Positive Definite**: no
- **Type**: binary
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: Laboratory for Web Algorithmics (LAW), Universita degli Studi di Milano, 
http://law.di.unimi.it/index.php.  When using matrices in the LAW/ group 
in the collection, please follow the citation instructions at            
http://law.di.unimi.it/datasets.php.  If you publish results based on    
these graphs, please acknowledge the usage of WebGraph and LLP by quoting
the following papers:                                                    
                                                                         
[1] "The WebGraph Framework I: Compression Techniques," Paolo Boldi      
    and Sebastiano Vigna, Proc. of the Thirteenth International          
    World Wide Web Conference (WWW 2004), 2004, Manhattan, USA,          
    pp. 595--601, ACM Press.                                             
                                                                         
[2] "Layered Label Propagation: A MultiResolution Coordinate-Free        
    Ordering for Compressing Social Networks," Paolo Boldi, Marco        
    Rosa, Massimo Santini, and Sebastiano Vigna, Proceedings of the      
    20th international conference on World Wide Web, 2011, ACM Press.    
                                                                         
If the graphs you are using were gathered by UbiCrawler, please          
acknowledge the usage of UbiCrawler by quoting the following paper:      
                                                                         
[3] "UbiCrawler: A Scalable Fully Distributed Web Crawler",              
    Paolo Boldi, Bruno Codenotti, Massimo Santini, and Sebastiano        
    Vigna, Software: Practice & Experience, 2004, vol 34, no. 8,         
    pp. 711--726                                                         
                                                                         
LAW/webbase-2001                                                         
                                                                         
This graph has been obtained from the 2001 crawl performed by the        
WebBase crawler. The data provided by WebBase has been filtered to       
eliminate invalid links and to normalise URLs. The experiments           
reported in reports "The WebGraph Framework I: Compression               
Techniques" and "Codes for the World-Wide Web", (both at                 
http://law.di.unimi.it/ ) are based on this graph and on                 
uk-2002. Note that for historical reasons the URLs of this               
graph are coded in ISO-8859-1.                                           
                                                                         
For additional graph properties and statistics, including node labels,   
see http://law.di.unimi.it/webdata/webbase-2001

---

## 8. lp_nug20

**Collection**: Qaplib  
**URL**: https://sparse.tamu.edu/Qaplib/lp_nug20

### Properties

- **Name**: lp_nug20
- **Group**: Qaplib
- **Matrix Id**: 799
- **Num Rows**: 15,240
- **Num Cols**: 72,600
- **Nonzeros**: 304,800
- **Pattern Entries**: 304,800
- **Kind**: Linear Programming Problem
- **Symmetric**: No
- **Date**: 1995
- **Author**: M. Resende
- **Editor**: M. Resende
- **Structural Rank**: 15,240
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 1
- **Strongly Connect Components**: 1
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 0%
- **Numeric Symmetry**: 0%
- **Cholesky Candidate**: no
- **Positive Definite**: no
- **Type**: integer
- **Matrix Norm**: 1.143457e+01
- **Minimum Singular Value**: 2.130423e-16
- **Condition Number**: 5.367278e+16
- **Rank**: 14,098
- **Sprank(A)-Rank(A)**: 1,142
- **Null Space Dimension**: 1,142
- **Full Numerical Rank?**: no
- **Download Singular Values**: MATLAB
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: 

---

## 9. ASIC_320k

**Collection**: Sandia  
**URL**: https://sparse.tamu.edu/Sandia/ASIC_320k

### Properties

- **Name**: ASIC_320k
- **Group**: Sandia
- **Matrix Id**: 1417
- **Num Rows**: 321,821
- **Num Cols**: 321,821
- **Nonzeros**: 1,931,828
- **Pattern Entries**: 2,635,364
- **Kind**: Circuit Simulation Problem
- **Symmetric**: No
- **Date**: 2006
- **Author**: R. Hoekstra
- **Editor**: T. Davis
- **Structural Rank**: 321,821
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 399
- **Strongly Connect Components**: 1
- **Num Explicit Zeros**: 703,536
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 35.8%
- **Cholesky Candidate**: no
- **Positive Definite**: no
- **Type**: real
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: 

---

## 10. ASIC_680k

**Collection**: Sandia  
**URL**: https://sparse.tamu.edu/Sandia/ASIC_680k

### Properties

- **Name**: ASIC_680k
- **Group**: Sandia
- **Matrix Id**: 1419
- **Num Rows**: 682,862
- **Num Cols**: 682,862
- **Nonzeros**: 2,638,997
- **Pattern Entries**: 3,871,773
- **Kind**: Circuit Simulation Problem
- **Symmetric**: No
- **Date**: 2006
- **Author**: R. Hoekstra
- **Editor**: T. Davis
- **Structural Rank**: 682,862
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 583,921
- **Strongly Connect Components**: 583,523
- **Num Explicit Zeros**: 1,232,776
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 0.4%
- **Cholesky Candidate**: no
- **Positive Definite**: no
- **Type**: real
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: 

---

## 11. webbase-1M

**Collection**: Williams  
**URL**: https://sparse.tamu.edu/Williams/webbase-1M

### Properties

- **Name**: webbase-1M
- **Group**: Williams
- **Matrix Id**: 2379
- **Num Rows**: 1,000,005
- **Num Cols**: 1,000,005
- **Nonzeros**: 3,105,536
- **Pattern Entries**: 3,105,536
- **Kind**: Weighted Directed Graph
- **Symmetric**: No
- **Date**: 2008
- **Author**: unknown
- **Editor**: S. Williams, L. Oliker, R. Vuduc, J. Shalf, K. Yelick, J. Demmel
- **Strongly Connect Components**: 940,398
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 9.6%
- **Numeric Symmetry**: 1.6%
- **Cholesky Candidate**: no
- **Positive Definite**: no
- **Type**: real
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: Matrices used by S. Williams et al for sparse matrix multiplication on GPUs.   
                                                                               
14 matrices were used in the following paper:                                  
                                                                               
    S. Williams, L. Oliker, R. Vuduc, J. Shalf, K. Yelick, J. Demmel,          
    "Optimization of Sparse Matrix-Vector Multiplication on Emerging Multicore 
    Platforms", Parallel Computing Volume 35, Issue 3, March 2009, Pages       
    178-194.  Special issue on Revolutionary Technologies for Acceleration of  
    Emerging Petascale Applications.                                           
                                                                               
    https://hpcrd.lbl.gov/~swwilliams/research/papers/parco08_spmv.pdf         
    http://dx.doi.org/10.1016/j.parco.2008.12.006                              
                                                                               
This same set of 14 matrices was also used in a subsequent technical report by 
NVIDIA:                                                                        
                                                                               
    http://www.nvidia.com/object/nvidia_research_pub_001.html "Efficient Sparse
    Matrix-Vector Multiplication on CUDA" Nathan Bell and Michael Garland, in, 
    "NVIDIA Technical Report NVR-2008-004", December 2008                      
                                                                               
file            Name            dim*    nnz     description                    
                                                                               
dense2          Dense           2K      4.0M    dense matrix in sparse format  
pdb1HYS         Protein         36K     4.3M    protein data bank 1HYS         
consph          FEM/Spheres     83K     6.0M    FEM concentric spheres         
cant            FEM/Cantilever  62K     4.0M    FEM cantilever                 
pwtk            Wind Tunnel     218K    11.6M   pressurized wind tunnel        
rma10           FEM/Harbor      47K     2.37M   3D CFD of Charleston Harbor    
qcd5_4          QCD             49K     1.90M   quark propagators (QCD/LGT)    
shipsec1        FEM/Ship        141K    3.98M   FEM Ship section / detail      
mac_econ_fwd500 Economics       207K    1.27M   Macroeconomic model            
mc2depi         Epidemiology    526K    2.1M    2D Markov model of epidemic    
cop20k_A        FEM/Accelerator 121K    2.62M   Accelerator cavity design      
scircuit        Circuit         171K    959K    Motorola circuit simulation    
webbase-1M      webbase         1M      3.1M    Web connectivity matrix        
rail4284        LP              4Kx1.1M 11.3M   Railways set cover,            
                                                constraint matrix              
                                                                               
(*) the matrix is square if only one dimension listed.                         
                                                                               
Six of the matrices are nearly identical to the matrices already in the        
UF Collection.  They are thus not included in the UF Collection.  See          
the README.txt file for this collection for details.

---

## 12. cnr-2000

**Collection**: LAW  
**URL**: https://sparse.tamu.edu/LAW/cnr-2000

### Properties

- **Name**: cnr-2000
- **Group**: LAW
- **Matrix Id**: 2441
- **Num Rows**: 325,557
- **Num Cols**: 325,557
- **Nonzeros**: 3,216,152
- **Pattern Entries**: 3,216,152
- **Kind**: Directed Graph
- **Symmetric**: No
- **Date**: 2000
- **Author**: Laboratory for Web Algorithmics (LAW), Universita degli Studi di Milano, http://law.di.unimi.it/index.php
- **Editor**: Laboratory for Web Algorithmics (LAW), Universita degli Studi di Milano, http://law.di.unimi.it/index.php
- **Strongly Connect Components**: 100,977
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 24.9%
- **Numeric Symmetry**: 24.9%
- **Cholesky Candidate**: no
- **Positive Definite**: no
- **Type**: binary
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: Laboratory for Web Algorithmics (LAW), Universita degli Studi di Milano, 
http://law.di.unimi.it/index.php.  When using matrices in the LAW/ group 
in the collection, please follow the citation instructions at            
http://law.di.unimi.it/datasets.php.  If you publish results based on    
these graphs, please acknowledge the usage of WebGraph and LLP by quoting
the following papers:                                                    
                                                                         
[1] "The WebGraph Framework I: Compression Techniques," Paolo Boldi      
    and Sebastiano Vigna, Proc. of the Thirteenth International          
    World Wide Web Conference (WWW 2004), 2004, Manhattan, USA,          
    pp. 595--601, ACM Press.                                             
                                                                         
[2] "Layered Label Propagation: A MultiResolution Coordinate-Free        
    Ordering for Compressing Social Networks," Paolo Boldi, Marco        
    Rosa, Massimo Santini, and Sebastiano Vigna, Proceedings of the      
    20th international conference on World Wide Web, 2011, ACM Press.    
                                                                         
If the graphs you are using were gathered by UbiCrawler, please          
acknowledge the usage of UbiCrawler by quoting the following paper:      
                                                                         
[3] "UbiCrawler: A Scalable Fully Distributed Web Crawler",              
    Paolo Boldi, Bruno Codenotti, Massimo Santini, and Sebastiano        
    Vigna, Software: Practice & Experience, 2004, vol 34, no. 8,         
    pp. 711--726                                                         
                                                                         
LAW/cnr-2000                                                             
                                                                         
A very small crawl of the Italian CNR domain.                            
                                                                         
For additional graph properties and statistics, including node labels,   
see http://law.di.unimi.it/webdata/cnr-2000

---

## 13. eu-2005

**Collection**: LAW  
**URL**: https://sparse.tamu.edu/LAW/eu-2005

### Properties

- **Name**: eu-2005
- **Group**: LAW
- **Matrix Id**: 2443
- **Num Rows**: 862,664
- **Num Cols**: 862,664
- **Nonzeros**: 19,235,140
- **Pattern Entries**: 19,235,140
- **Kind**: Directed Graph
- **Symmetric**: No
- **Date**: 2005
- **Author**: Laboratory for Web Algorithmics (LAW), Universita degli Studi di Milano, http://law.di.unimi.it/index.php
- **Editor**: Laboratory for Web Algorithmics (LAW), Universita degli Studi di Milano, http://law.di.unimi.it/index.php
- **Strongly Connect Components**: 90,768
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 27.7%
- **Numeric Symmetry**: 27.7%
- **Cholesky Candidate**: no
- **Positive Definite**: no
- **Type**: binary
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: Laboratory for Web Algorithmics (LAW), Universita degli Studi di Milano, 
http://law.di.unimi.it/index.php.  When using matrices in the LAW/ group 
in the collection, please follow the citation instructions at            
http://law.di.unimi.it/datasets.php.  If you publish results based on    
these graphs, please acknowledge the usage of WebGraph and LLP by quoting
the following papers:                                                    
                                                                         
[1] "The WebGraph Framework I: Compression Techniques," Paolo Boldi      
    and Sebastiano Vigna, Proc. of the Thirteenth International          
    World Wide Web Conference (WWW 2004), 2004, Manhattan, USA,          
    pp. 595--601, ACM Press.                                             
                                                                         
[2] "Layered Label Propagation: A MultiResolution Coordinate-Free        
    Ordering for Compressing Social Networks," Paolo Boldi, Marco        
    Rosa, Massimo Santini, and Sebastiano Vigna, Proceedings of the      
    20th international conference on World Wide Web, 2011, ACM Press.    
                                                                         
If the graphs you are using were gathered by UbiCrawler, please          
acknowledge the usage of UbiCrawler by quoting the following paper:      
                                                                         
[3] "UbiCrawler: A Scalable Fully Distributed Web Crawler",              
    Paolo Boldi, Bruno Codenotti, Massimo Santini, and Sebastiano        
    Vigna, Software: Practice & Experience, 2004, vol 34, no. 8,         
    pp. 711--726                                                         
                                                                         
LAW/eu-2005                                                              
                                                                         
A small crawl of the .eu domain.  This graph exhibits                    
a very low locality, probably because the crawl was quite                
shallow (and the chosen domain is quite artificial anyway).              
                                                                         
For additional graph properties and statistics, including node labels,   
see http://law.di.unimi.it/webdata/eu-2005

---

## 14. in-2004

**Collection**: LAW  
**URL**: https://sparse.tamu.edu/LAW/in-2004

### Properties

- **Name**: in-2004
- **Group**: LAW
- **Matrix Id**: 2442
- **Num Rows**: 1,382,908
- **Num Cols**: 1,382,908
- **Nonzeros**: 16,917,053
- **Pattern Entries**: 16,917,053
- **Kind**: Directed Graph
- **Symmetric**: No
- **Date**: 2004
- **Author**: Laboratory for Web Algorithmics (LAW), Universita degli Studi di Milano, http://law.di.unimi.it/index.php
- **Editor**: Laboratory for Web Algorithmics (LAW), Universita degli Studi di Milano, http://law.di.unimi.it/index.php
- **Strongly Connect Components**: 367,675
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 35.6%
- **Numeric Symmetry**: 35.6%
- **Cholesky Candidate**: no
- **Positive Definite**: no
- **Type**: binary
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: Laboratory for Web Algorithmics (LAW), Universita degli Studi di Milano, 
http://law.di.unimi.it/index.php.  When using matrices in the LAW/ group 
in the collection, please follow the citation instructions at            
http://law.di.unimi.it/datasets.php.  If you publish results based on    
these graphs, please acknowledge the usage of WebGraph and LLP by quoting
the following papers:                                                    
                                                                         
[1] "The WebGraph Framework I: Compression Techniques," Paolo Boldi      
    and Sebastiano Vigna, Proc. of the Thirteenth International          
    World Wide Web Conference (WWW 2004), 2004, Manhattan, USA,          
    pp. 595--601, ACM Press.                                             
                                                                         
[2] "Layered Label Propagation: A MultiResolution Coordinate-Free        
    Ordering for Compressing Social Networks," Paolo Boldi, Marco        
    Rosa, Massimo Santini, and Sebastiano Vigna, Proceedings of the      
    20th international conference on World Wide Web, 2011, ACM Press.    
                                                                         
If the graphs you are using were gathered by UbiCrawler, please          
acknowledge the usage of UbiCrawler by quoting the following paper:      
                                                                         
[3] "UbiCrawler: A Scalable Fully Distributed Web Crawler",              
    Paolo Boldi, Bruno Codenotti, Massimo Santini, and Sebastiano        
    Vigna, Software: Practice & Experience, 2004, vol 34, no. 8,         
    pp. 711--726                                                         
                                                                         
LAW/in-2004                                                              
                                                                         
A small crawl of the .in domain performed for the Nagaoka                
University of Technology.                                                
                                                                         
For additional graph properties and statistics, including node labels,   
see http://law.di.unimi.it/webdata/in-2004

---

## 15. thermomech_dK

**Collection**: Botonakis  
**URL**: https://sparse.tamu.edu/Botonakis/thermomech_dK

### Properties

- **Name**: thermomech_dK
- **Group**: Botonakis
- **Matrix Id**: 2260
- **Num Rows**: 204,316
- **Num Cols**: 204,316
- **Nonzeros**: 2,846,228
- **Pattern Entries**: 2,846,228
- **Kind**: Thermal Problem
- **Symmetric**: No
- **Date**: 2009
- **Author**: I. Botonakis
- **Editor**: T. Davis
- **Structural Rank**: 204,316
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 1
- **Strongly Connect Components**: 1
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 66.5%
- **Cholesky Candidate**: no
- **Positive Definite**: no
- **Type**: real
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: TC and TK are for temperature, dM and dK for deformation

---

## 16. ldoor

**Collection**: GHS_psdef  
**URL**: https://sparse.tamu.edu/GHS_psdef/ldoor

### Properties

- **Name**: ldoor
- **Group**: GHS_psdef
- **Matrix Id**: 1268
- **Num Rows**: 952,203
- **Num Cols**: 952,203
- **Nonzeros**: 42,493,817
- **Pattern Entries**: 46,522,475
- **Kind**: Structural Problem
- **Symmetric**: Yes
- **Date**: 2004
- **Author**: J. Weiher
- **Editor**: J. Koster
- **Structural Rank**: 952,203
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 42,667
- **Strongly Connect Components**: 42,667
- **Num Explicit Zeros**: 4,028,658
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 100%
- **Cholesky Candidate**: yes
- **Positive Definite**: yes
- **Type**: real
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: 

---

## 17. mip1

**Collection**: Andrianov  
**URL**: https://sparse.tamu.edu/Andrianov/mip1

### Properties

- **Name**: mip1
- **Group**: Andrianov
- **Matrix Id**: 1385
- **Num Rows**: 66,463
- **Num Cols**: 66,463
- **Nonzeros**: 10,352,819
- **Pattern Entries**: 10,352,819
- **Kind**: Optimization Problem
- **Symmetric**: Yes
- **Date**: 2006
- **Author**: A. Andrianov
- **Editor**: T. Davis
- **Structural Rank**: 66,463
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 1
- **Strongly Connect Components**: 1
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 100%
- **Cholesky Candidate**: yes
- **Positive Definite**: no
- **Type**: binary
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: 

---

## 18. hvdc2

**Collection**: HVDC  
**URL**: https://sparse.tamu.edu/HVDC/hvdc2

### Properties

- **Name**: hvdc2
- **Group**: HVDC
- **Matrix Id**: 1875
- **Num Rows**: 189,860
- **Num Cols**: 189,860
- **Nonzeros**: 1,339,638
- **Pattern Entries**: 1,347,273
- **Kind**: Power Network Problem
- **Symmetric**: No
- **Date**: 2007
- **Author**: A. Wang
- **Editor**: T. Davis
- **Structural Rank**: 189,860
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 69
- **Strongly Connect Components**: 67
- **Num Explicit Zeros**: 7,635
- **Pattern Symmetry**: 98.9%
- **Numeric Symmetry**: 6.3%
- **Cholesky Candidate**: no
- **Positive Definite**: no
- **Type**: real
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: 

---

## 19. fullb

**Collection**: DNVS  
**URL**: https://sparse.tamu.edu/DNVS/fullb

### Properties

- **Name**: fullb
- **Group**: DNVS
- **Matrix Id**: 1264
- **Num Rows**: 199,187
- **Num Cols**: 199,187
- **Nonzeros**: 11,708,077
- **Pattern Entries**: 11,708,077
- **Kind**: Structural Problem
- **Symmetric**: Yes
- **Date**: 2004
- **Author**: C. Damhaug
- **Editor**: N. Gould, Y. Hu, J. Scott
- **Structural Rank**: 199,187
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 1
- **Strongly Connect Components**: 1
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 100%
- **Cholesky Candidate**: yes
- **Positive Definite**: no
- **Type**: binary
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: 

---

## 20. ins2

**Collection**: Andrianov  
**URL**: https://sparse.tamu.edu/Andrianov/ins2

### Properties

- **Name**: ins2
- **Group**: Andrianov
- **Matrix Id**: 1382
- **Num Rows**: 309,412
- **Num Cols**: 309,412
- **Nonzeros**: 2,751,484
- **Pattern Entries**: 2,751,484
- **Kind**: Optimization Problem
- **Symmetric**: Yes
- **Date**: 2006
- **Author**: A. Andrianov
- **Editor**: T. Davis
- **Structural Rank**: 309,412
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 1
- **Strongly Connect Components**: 1
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 100%
- **Cholesky Candidate**: yes
- **Positive Definite**: no
- **Type**: binary
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: 

---

## 21. Ga41As41H72

**Collection**: PARSEC  
**URL**: https://sparse.tamu.edu/PARSEC/Ga41As41H72

### Properties

- **Name**: Ga41As41H72
- **Group**: PARSEC
- **Matrix Id**: 1353
- **Num Rows**: 268,096
- **Num Cols**: 268,096
- **Nonzeros**: 18,488,476
- **Pattern Entries**: 18,488,476
- **Kind**: Theoretical/Quantum Chemistry Problem
- **Symmetric**: Yes
- **Date**: 2005
- **Author**: Y. Zhou, Y. Saad, M. Tiago, J. Chelikowsky
- **Editor**: T. Davis
- **Structural Rank**: 268,096
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 1
- **Strongly Connect Components**: 1
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 100%
- **Cholesky Candidate**: yes
- **Positive Definite**: no
- **Type**: real
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: 

---

## 22. Si41Ge41H72

**Collection**: PARSEC  
**URL**: https://sparse.tamu.edu/PARSEC/Si41Ge41H72

### Properties

- **Name**: Si41Ge41H72
- **Group**: PARSEC
- **Matrix Id**: 1362
- **Num Rows**: 185,639
- **Num Cols**: 185,639
- **Nonzeros**: 15,011,265
- **Pattern Entries**: 15,011,265
- **Kind**: Theoretical/Quantum Chemistry Problem
- **Symmetric**: Yes
- **Date**: 2005
- **Author**: Y. Zhou, Y. Saad, M. Tiago, J. Chelikowsky
- **Editor**: T. Davis
- **Structural Rank**: 185,639
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 1
- **Strongly Connect Components**: 1
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 100%
- **Cholesky Candidate**: yes
- **Positive Definite**: no
- **Type**: real
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: 

---

## 23. shipsec5

**Collection**: DNVS  
**URL**: https://sparse.tamu.edu/DNVS/shipsec5

### Properties

- **Name**: shipsec5
- **Group**: DNVS
- **Matrix Id**: 1280
- **Num Rows**: 179,860
- **Num Cols**: 179,860
- **Nonzeros**: 4,598,604
- **Pattern Entries**: 10,113,096
- **Kind**: Structural Problem
- **Symmetric**: Yes
- **Date**: 1999
- **Author**: C. Damhaug
- **Editor**: J. Koster
- **Structural Rank**: 179,860
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 36
- **Strongly Connect Components**: 36
- **Num Explicit Zeros**: 5,514,492
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 100%
- **Cholesky Candidate**: yes
- **Positive Definite**: yes
- **Type**: real
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: 

---

## 24. windscreen

**Collection**: Oberwolfach  
**URL**: https://sparse.tamu.edu/Oberwolfach/windscreen

### Properties

- **Name**: windscreen
- **Group**: Oberwolfach
- **Matrix Id**: 1452
- **Num Rows**: 22,692
- **Num Cols**: 22,692
- **Nonzeros**: 1,482,390
- **Pattern Entries**: 1,482,390
- **Kind**: Model Reduction Problem
- **Symmetric**: Yes
- **Date**: 2004
- **Author**: K. Meerbergen
- **Editor**: E. Rudnyi
- **Structural Rank**: 22,692
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 1
- **Strongly Connect Components**: 1
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 0%
- **Cholesky Candidate**: no
- **Positive Definite**: no
- **Type**: complex
- **Matrix Norm**: 1.887078e+11
- **Minimum Singular Value**: 1.778172e-07
- **Condition Number**: 1.061246e+18
- **Rank**: 22,686
- **Sprank(A)-Rank(A)**: 6
- **Null Space Dimension**: 6
- **Full Numerical Rank?**: no
- **Download Singular Values**: MATLAB
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: Primary matrix in this model reduction problem is the Oberwolfach K matrix

---

## 25. thermal2

**Collection**: Schmid  
**URL**: https://sparse.tamu.edu/Schmid/thermal2

### Properties

- **Name**: thermal2
- **Group**: Schmid
- **Matrix Id**: 1403
- **Num Rows**: 1,228,045
- **Num Cols**: 1,228,045
- **Nonzeros**: 8,580,313
- **Pattern Entries**: 8,580,313
- **Kind**: Thermal Problem
- **Symmetric**: Yes
- **Date**: 2006
- **Author**: D. Schmid
- **Editor**: T. Davis
- **Structural Rank**: 1,228,045
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 959
- **Strongly Connect Components**: 959
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 100%
- **Cholesky Candidate**: yes
- **Positive Definite**: yes
- **Type**: real
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: 

---

## 26. offshore

**Collection**: Um  
**URL**: https://sparse.tamu.edu/Um/offshore

### Properties

- **Name**: offshore
- **Group**: Um
- **Matrix Id**: 2283
- **Num Rows**: 259,789
- **Num Cols**: 259,789
- **Nonzeros**: 4,242,673
- **Pattern Entries**: 4,242,673
- **Kind**: Electromagnetics Problem
- **Symmetric**: Yes
- **Date**: 2010
- **Author**: E. Um
- **Editor**: T. Davis
- **Structural Rank**: 259,789
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 1
- **Strongly Connect Components**: 1
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 100%
- **Cholesky Candidate**: yes
- **Positive Definite**: yes
- **Type**: real
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: This is a finite-element system matrix deriving from the 3D transient
electric field diffusion equation. The matrix results from the       
discretization of offshore conductivity structures using tetrahedral 
elements.  Evan Um, Geophysics, Stanford.

---

## 27. poisson3Db

**Collection**: FEMLAB  
**URL**: https://sparse.tamu.edu/FEMLAB/poisson3Db

### Properties

- **Name**: poisson3Db
- **Group**: FEMLAB
- **Matrix Id**: 928
- **Num Rows**: 85,623
- **Num Cols**: 85,623
- **Nonzeros**: 2,374,949
- **Pattern Entries**: 2,374,949
- **Kind**: Computational Fluid Dynamics Problem
- **Symmetric**: No
- **Date**: 2003
- **Author**: COMSOL
- **Editor**: T. Davis
- **Structural Rank**: 85,623
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 1
- **Strongly Connect Components**: 1
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 0%
- **Cholesky Candidate**: no
- **Positive Definite**: no
- **Type**: real
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: 

---

## 28. Hook_1498

**Collection**: Janna  
**URL**: https://sparse.tamu.edu/Janna/Hook_1498

### Properties

- **Name**: Hook_1498
- **Group**: Janna
- **Matrix Id**: 2546
- **Num Rows**: 1,498,023
- **Num Cols**: 1,498,023
- **Nonzeros**: 59,374,451
- **Pattern Entries**: 60,917,445
- **Kind**: Structural Problem
- **Symmetric**: Yes
- **Date**: 2011
- **Author**: C. Janna, M. Ferronato
- **Editor**: T. Davis
- **Structural Rank**: 1,498,023
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 30,001
- **Strongly Connect Components**: 30,001
- **Num Explicit Zeros**: 1,542,994
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 100%
- **Cholesky Candidate**: yes
- **Positive Definite**: yes
- **Type**: real
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: Authors: Carlo Janna and Massimiliano Ferronato                
                                                               
Symmetric Positive Definite Matrix                             
# equations:   1498023                                         
# non-zeroes: 60917445                                         
Problem description: Structural problem                        
                                                               
The matrix Hook_1498 is obtained from a 3D mechanical problem  
discretizing a steel hook with tetrahedral Finite Elements. The
computational grid consists of regularly shaped elements with  
three displacement unknowns associated to each node.

---

## 29. nasasrb

**Collection**: Nasa  
**URL**: https://sparse.tamu.edu/Nasa/nasasrb

### Properties

- **Name**: nasasrb
- **Group**: Nasa
- **Matrix Id**: 761
- **Num Rows**: 54,870
- **Num Cols**: 54,870
- **Nonzeros**: 2,677,324
- **Pattern Entries**: 2,677,324
- **Kind**: Structural Problem
- **Symmetric**: Yes
- **Date**: 1995
- **Author**: NASA
- **Editor**: G. Kumfert, A. Pothen
- **Structural Rank**: 54,870
- **Structural Rank Full**: true
- **Num Dmperm Blocks**: 1
- **Strongly Connect Components**: 1
- **Num Explicit Zeros**: 0
- **Pattern Symmetry**: 100%
- **Numeric Symmetry**: 100%
- **Cholesky Candidate**: yes
- **Positive Definite**: yes
- **Type**: real
- **Download**: MATLAB

Rutherford Boeing

Matrix Market
- **Notes**: 

---
