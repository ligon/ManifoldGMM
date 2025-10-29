;;; ob-fold-hidden.el --- Smart selective folding of Org src blocks -*- lexical-binding: t; -*-
;; Interpret :hidden yes/no relative to #+STARTUP: hideblocks
;;
;; Copyright (C) 2025 Ethan Ligon
;;
;; Author: Ethan Ligon <ligon@berkeley.edu>
;; Maintainer: Ethan Ligon <ligon@berkeley.edu>
;; Created: October 28, 2025
;; Modified: October 28, 2025
;; Version: 0.0.2
;; Keywords: orgmode organization
;; Package-Requires: ((emacs "28.1") (org "9.4"))
;; This file is not part of GNU Emacs.
;;
;;; Commentary:
;;
;;  Smart selective folding of Org-mode source blocks, honoring a `:hidden`
;;  header arg as well as the global hideblocks behavior.
;;
;;  Install in your project and call `el/org-setup-fold-hidden` on Org file
;;  open for auto-selective folding!
;;
;;; Code:

(require 'cl-lib)
(require 'ob-core)

(defun el/get-org-hideblocks-startup ()
  "Return non-nil if #+STARTUP: hideblocks is in effect in current buffer."
  (or org-hide-block-startup
      (save-excursion
        (let ((case-fold-search t))
          (goto-char (point-min))
          (re-search-forward
           "^[ \t]*#\\+startup:.*\\bhideblocks\\b" nil t)))))

(defun el/normalize-param (value)
  "Return VALUE as a lower-case string when possible."
  (cond
   ((stringp value) (downcase value))
   ((symbolp value) (downcase (symbol-name value)))
   ((numberp value) (downcase (number-to-string value)))
   (t value)))

(defun el/string-case-equal (a b)
  "Case-insensitive string comparison."
  (let ((sa (el/normalize-param a))
        (sb (el/normalize-param b)))
    (and (stringp sa) (stringp sb)
         (string-equal sa sb))))

(defun el/fold-selected-src-blocks ()
  "Hide or show Org src blocks per :hidden, :exports, and `#+STARTUP: hideblocks`.
- With global hideblocks, hide all src blocks unless :hidden no.
- Otherwise, show all, except :hidden yes or :exports none.
:exports none always hides block content."
  (interactive)
  (when (derived-mode-p 'org-mode)
    (let ((hideblocks (el/get-org-hideblocks-startup)))
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
                 (should-hide
                  (cond
                   ;; :hidden yes (explicit hide)
                   ((el/string-case-equal hidden "yes") t)
                   ;; :hidden no (explicit don't hide)
                   ((el/string-case-equal hidden "no") nil)
                   ;; global hideblocks: hide unless :hidden no
                   (hideblocks t)
                   ;; otherwise: hide if :hidden yes or :exports none
                   (t (or (el/string-case-equal hidden "yes")
                          (el/string-case-equal exports "none")))))
                 ;; Is this block currently folded (hidden)?
                 (folded
                  (let ((end (save-excursion
                               (goto-char beg)
                               (org-babel-end-of-src-block)
                               (point))))
                    (save-excursion
                      (goto-char (1+ beg))
                      (invisible-p (point))))))
            (goto-char beg)
            ;; Only toggle if there's a state change needed!
            ;; Use org-hide-block-toggle, as org-babel-hide-block-toggle is deprecated
            (when (cl-xor should-hide folded)
              (org-hide-block-toggle (if should-hide t 'off) t))))))))

(defun el/org-setup-fold-hidden ()
  "Set up selective code folding in current Org buffer.
Disables buffer-local auto-global fold, then runs `el/fold-selected-src-blocks'."
  (setq-local org-hide-block-startup nil)
  (el/fold-selected-src-blocks))

(provide 'ob-fold-hidden)
;;; ob-fold-hidden.el ends here
