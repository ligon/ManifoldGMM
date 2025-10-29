;;; ob-fold-hidden.el --- Smart selective folding of Org src blocks -*- lexical-binding: t; -*-
;; Interpret :hidden yes/no relative to #+STARTUP: hideblocks
;;
;; Copyright (C) 2025 Ethan Ligon
;;
;; Author: Ethan Ligon <ligon@berkeley.edu>
;; Maintainer: Ethan Ligon <ligon@berkeley.edu>
;; Created: October 28, 2025
;; Modified: October 28, 2025
;; Version: 0.0.1
;; Keywords: orgmode organization
;; Package-Requires: ((emacs "27.1") (org "9.4"));;
;; This file is not part of GNU Emacs.
;;
;; Package-Requires: ((emacs "27.1") (org "9.4"))
;;
;;; Commentary:
;;
;;  This file defines helper functions that provide *smart selective folding*
;;  of Org-mode source blocks.  It extends the basic behavior of
;;  `#+STARTUP: hideblocks` by allowing per-block overrides through a custom
;;  header argument `:hidden`.
;;
;;  Behavior summary:
;;
;;    - When `#+STARTUP: hideblocks` (or `org-hide-block-startup` non-nil),
;;      all source blocks are hidden by default *except* those with
;;      `:hidden no`.
;;
;;    - When hideblocks is *not* in effect, source blocks are shown by
;;      default, except those with `:hidden yes` or `:exports none`.
;;
;;    - The parameter `:hidden yes` always forces a block to be hidden.
;;      The parameter `:hidden no` always forces a block to remain visible.
;;
;;  This allows authors to maintain clean pedagogical Org documents in which
;;  verbose plotting or data-prep code is quietly folded, while concise
;;  illustrative code remains visible—even when `hideblocks` is globally on.
;;
;;  Typical setup (in a project `.dir-locals.el`):
;;
;;    ((org-mode
;;      . ((eval . (progn
;;                   (load-file (expand-file-name "tools/ob-fold-hidden.el"
;;                                                (locate-dominating-file default-directory ".dir-locals.el")))
;;                   (add-hook 'find-file-hook #'el/org-setup-fold-hidden nil t))))))
;;
;;  Once this file is loaded, opening any Org buffer in the project will
;;  automatically fold or unfold blocks according to the rules above.
;;
;;; Code:


(defun el/fold-selected-src-blocks ()
  "Fold or unfold Org src blocks depending on their :hidden param and global hideblocks.
If #+STARTUP: hideblocks (org-hide-block-startup) is non-nil, hide all by default
*except* blocks marked ':hidden no'.
If hideblocks is nil, show all by default *except* blocks marked ':hidden yes'
or with ':exports none'."
  (interactive)
  (when (derived-mode-p 'org-mode)
    (save-excursion
      (goto-char (point-min))
      (while (re-search-forward org-babel-src-block-regexp nil t)
        (let* ((beg (match-beginning 0))
               (info (save-excursion
                       (goto-char beg)
                       (org-babel-get-src-block-info 'light)))
               (params (nth 2 info))
               (hidden (alist-get :hidden params))
               (exports (alist-get :exports params))
               ;; Determine whether hideblocks is in effect.
               ;; Use `org-hide-block-startup` if non-nil, otherwise look for a
               ;; `#+STARTUP:` or `#+startup:` line containing the token `hideblocks`
               ;; (case-insensitive).
               (hideblocks
                (or org-hide-block-startup
                    (save-excursion
                      (goto-char (point-min))
                      (re-search-forward
                       "^[ \t]*#\\+startup:.*\\bhideblocks\\b" nil t 'case-insensitive))))
               ;; decide whether to hide this one:
               (should-hide
                (cond
                 ;; explicit override
                 ((eq hidden 'yes) t)
                 ((eq hidden 'no) nil)
                 ;; global hideblocks in effect → hide unless :hidden no
                 (hideblocks (not (eq hidden 'no)))
                 ;; global not hiding → hide only :hidden yes or exports none
                 (t (or (eq hidden 'yes)
                        (and (stringp exports)
                             (string= exports "none")))))))
          (goto-char beg)
          (org-babel-hide-block-toggle (if should-hide 'hide 'show))))))))

(defun el/org-setup-fold-hidden ()
  "Apply `el/fold-selected-src-blocks' buffer-locally after opening Org file."
  (setq-local org-hide-block-startup nil) ;; don’t auto-hide everything
  (el/fold-selected-src-blocks))

(provide 'ob-fold-hidden)
;;; ob-fold-hidden.el ends here

